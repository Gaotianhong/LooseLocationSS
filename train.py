import os
import ast
import copy
import time
import torch
import logging
import argparse
import traceback

import albumentations as A
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.crc_model import CRCModel
from dataset import CRCDataset, CRCPatchDataset
from utils.losses import FuzzyClassificationLoss, js_divergence
from utils.utils import set_seed, evaluate_model, plot_history, get_region, generate_mask


set_seed(seed=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, criterion, criterion_fuzzy):

    best_val = None
    train_epoch_loss, train_epoch_auc, train_epoch_precision, train_epoch_sensitivity, train_epoch_specificity, train_epoch_f1score = [], [], [], [], [], []
    val_epoch_loss, val_epoch_auc, val_epoch_precision, val_epoch_sensitivity, val_epoch_specificity, val_epoch_f1score = [], [], [], [], [], []

    # dataloader len
    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)

    alpha = 0.5  # loss cls
    beta = 1  # loss con
    gamma = 0.5  # loss loop

    ratio = 0.25  # annotations scale, 1/3 modality * ratio
    cell_size = 32  # grid cell size

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.epochs)

    for epoch in range(args.epochs):

        model.train()
        running_loss = 0
        y_true, y_scores, y_pred = [], [], []
        for i, data in enumerate(train_loader, 0):
            inputs, volumes, boxes, labels = data
            inputs = inputs.to(device)
            volumes = volumes.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # loss cls
            loss_cls = criterion(outputs, labels)

            # loss patch
            loss_neg_patch_cls = torch.tensor(0)  # negative patch cls
            loss_pos_patch_cls_sup = torch.tensor(0)  # positive patch cls
            loss_pos_patch_cls_unsup = torch.tensor(0)  # positive patch cls
            loss_patch = torch.tensor(0)
            # loss temporal and loss modality
            loss_tm = torch.tensor(0)
            # loss con
            loss_con = torch.tensor(0)

            # loss loop
            loss_loop = torch.tensor(0)

            if args.p_cls:  # patch cls
                outputs, outputs_p_cls, outputsnei_p_cls = model(inputs, volumes)
            else:  # image cls
                outputs = model(inputs, volumes)

            # lesion slices
            mask_pos = ~torch.all(boxes == torch.tensor([-1.0, -1.0, -1.0, -1.0]).to(device), dim=1)

            if args.p_cls:
                # neg patch cls
                filter_neg_logit = outputs_p_cls[labels == 0]
                filter_neg_logit = filter_neg_logit.reshape(filter_neg_logit.shape[0], 2, -1).permute(0, 2, 1).reshape(-1, 2)
                loss_neg_patch_cls = criterion(filter_neg_logit, torch.zeros_like(labels[labels == 0]).repeat(7*7))

                # pos patch cls with sup
                outputs_pos_p_cls = outputs_p_cls[mask_pos]
                sup_boxes = boxes[mask_pos].cpu().detach().numpy() * 7
                len_sup = int(len(sup_boxes) * ratio)
                patch_labels = torch.zeros((len_sup, 7, 7)).to(device)
                for j in range(len_sup):
                    yy, xx = int(sup_boxes[j][0]), int(sup_boxes[j][1])
                    patch_labels[j][yy][xx] = 1
                filter_pos_logit = outputs_pos_p_cls[:len_sup].reshape(len_sup, 2, -1).permute(0, 2, 1).reshape(-1, 2)
                loss_pos_patch_cls_sup = criterion(filter_pos_logit, patch_labels.view(-1).long())

                # pos patch cls unsup
                loss_pos_patch_cls_unsup = criterion_fuzzy(outputs_pos_p_cls[len_sup:])
                loss_patch = loss_pos_patch_cls_unsup + loss_pos_patch_cls_sup + loss_neg_patch_cls

                # temporal and modality consistency
                outputs_p_cls_prob = F.softmax(outputs_p_cls[labels == 1], dim=1)
                mask_fg = generate_mask(outputs_p_cls_prob)
                outputs_p_cls_prob = outputs_p_cls_prob * mask_fg
                for j in range(4):
                    outputsnei_logit = outputsnei_p_cls[:, j, :, :, :]
                    outputsnei_prob = F.softmax(outputsnei_logit[labels == 1], dim=1) * mask_fg

                    p = outputsnei_prob.reshape(outputsnei_prob.shape[0], 2, -1).permute(0, 2, 1).reshape(-1, 2)
                    q = outputs_p_cls_prob.reshape(outputs_p_cls_prob.shape[0], 2, -1).permute(0, 2, 1).reshape(-1, 2)
                    js_con = js_divergence(p, q).mean()  # js divergence
                    loss_tm += js_con.float()
                loss_tm /= 4

                # loss con
                loss_con = loss_patch + loss_tm

                # loopback
                inputs_pos = inputs[mask_pos]
                for j in range(len(inputs_pos)):
                    loop_mask = torch.ones(inputs_pos[j].shape).to(device)
                    yy, xx = get_region(outputs_pos_p_cls[j].unsqueeze(0))
                    loop_mask[:, int(yy*cell_size):int((yy+1)*cell_size), int(xx*cell_size):int((xx+1)*cell_size)] = 0
                    inputs_pos[j] = inputs_pos[j] * loop_mask
                outputs_mask = model(inputs_pos, volumes, loopback=True)
                loss_loop = criterion(outputs_mask, torch.zeros(len(inputs_pos)).long().to(device))

                loss = alpha * loss_cls + beta * loss_con + gamma * loss_loop
            else:
                # classification loss
                loss = loss_cls

            loss.backward()
            optimizer.step()

            Y_prob = F.softmax(outputs, dim=1)
            Y_hat = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().detach().numpy())
            y_scores.extend(Y_prob[:, 1].cpu().detach().numpy())
            y_pred.extend(Y_hat.cpu().detach().numpy())

            running_loss += loss.item() * inputs.size(0)

            # print
            if i % args.log_step == 0:
                auc, precision, sensitivity, specificity, f1score = evaluate_model(labels.cpu().detach().numpy(),
                                                                                   Y_prob[:, 1].cpu().detach().numpy(),
                                                                                   Y_hat.cpu().detach().numpy())
                logging.info(
                    "Training epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, train_loader_len) +
                    "lr:{:.8f} ".format(optimizer.param_groups[0]['lr']) +
                    "auc:{:.3f} ".format(auc) +
                    "precision:{:.3f} ".format(precision) +
                    "sensitivity:{:.3f} ".format(sensitivity) +
                    "specificity:{:.3f} ".format(specificity) +
                    "f1_score:{:.3f} ".format(f1score) +
                    "loss_cls:{:.3f} ".format(loss_cls.item()) +
                    "loss_patch:{:.3f} ".format(loss_patch.item()) +
                    "loss_tm:{:.3f} ".format(loss_tm.item()) +
                    "loss_con:{:.3f} ".format(loss_con.item()) +
                    "loss_loop:{:.3f} ".format(loss_loop.item()) +
                    "loss:{:.3f}".format(loss.item())
                )

        scheduler.step()

        model.eval()
        val_loss = 0
        val_y_true, val_y_scores, val_y_pred = [], [], []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, volumes, boxes, labels = data
                inputs = inputs.to(device)
                volumes = volumes.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device)

                if args.p_cls:  # patch cls
                    outputs, outputs_p_cls, outputsnei_p_cls = model(inputs, volumes)
                else:  # image cls
                    outputs = model(inputs, volumes)

                Y_prob = F.softmax(outputs, dim=1)
                Y_hat = torch.argmax(outputs, dim=1)

                # loss cls
                loss_cls = criterion(outputs, labels)

                # loss patch
                loss_neg_patch_cls = torch.tensor(0)
                loss_pos_patch_cls_sup = torch.tensor(0)
                loss_pos_patch_cls_unsup = torch.tensor(0)
                loss_patch = torch.tensor(0)
                # loss temporal & loss modality
                loss_tm = torch.tensor(0)
                # loss con
                loss_con = torch.tensor(0)

                # loss loop
                loss_loop = torch.tensor(0)

                mask_pos = ~torch.all(boxes == torch.tensor([-1.0, -1.0, -1.0, -1.0]).to(device), dim=1)
                if args.p_cls:
                    # neg patch cls
                    filter_neg_logit = outputs_p_cls[labels == 0]
                    filter_neg_logit = filter_neg_logit.reshape(filter_neg_logit.shape[0], 2, -1).permute(0, 2, 1).reshape(-1, 2)
                    loss_neg_patch_cls = criterion(filter_neg_logit, torch.zeros_like(labels[labels == 0]).repeat(7*7))
                    outputs_pos_p_cls = outputs_p_cls[mask_pos]
                    # pos patch cls unsup
                    loss_pos_patch_cls_unsup = criterion_fuzzy(outputs_pos_p_cls[len_sup:])
                    # pos patch cls sup
                    sup_boxes = boxes[mask_pos].cpu().detach().numpy() * 7
                    len_sup = int(len(sup_boxes) * ratio)
                    patch_labels = torch.zeros((len_sup, 7, 7)).to(device)
                    for j in range(len_sup):
                        yy, xx = int(sup_boxes[j][0]), int(sup_boxes[j][1])
                        patch_labels[j][yy][xx] = 1
                    filter_pos_logit = outputs_pos_p_cls[:len_sup].reshape(len_sup, 2, -1).permute(0, 2, 1).reshape(-1, 2)
                    loss_pos_patch_cls_sup = criterion(filter_pos_logit, patch_labels.view(-1).long())

                    loss_patch = loss_pos_patch_cls_unsup + loss_pos_patch_cls_sup + loss_neg_patch_cls

                    # temporal and modality consistency
                    outputs_p_cls_prob = F.softmax(outputs_p_cls[labels == 1], dim=1)
                    mask_fg = generate_mask(outputs_p_cls_prob)
                    outputs_p_cls_prob = outputs_p_cls_prob * mask_fg
                    for j in range(4):
                        outputsnei_logit = outputsnei_p_cls[:, j, :, :, :]
                        outputsnei_prob = F.softmax(outputsnei_logit[labels == 1], dim=1) * mask_fg

                        p = outputsnei_prob.reshape(outputsnei_prob.shape[0], 2, -1).permute(0, 2, 1).reshape(-1, 2)
                        q = outputs_p_cls_prob.reshape(outputs_p_cls_prob.shape[0], 2, -1).permute(0, 2, 1).reshape(-1, 2)
                        js_con = js_divergence(p, q).mean()
                        loss_tm += js_con.float()
                    loss_tm /= 4
                    # loss con
                    loss_con = loss_patch + loss_tm

                    # loopback
                    inputs_pos = inputs[mask_pos]
                    for j in range(len(inputs_pos)):
                        loop_mask = torch.ones(inputs_pos[j].shape).to(device)
                        yy, xx = get_region(outputs_pos_p_cls[j].unsqueeze(0))
                        loop_mask[:, int(yy*cell_size):int((yy+1)*cell_size), int(xx*cell_size):int((xx+1)*cell_size)] = 0
                        inputs_pos[j] = inputs_pos[j] * loop_mask
                    outputs_mask = model(inputs_pos, volumes, loopback=True)
                    loss_loop = criterion(outputs_mask, torch.zeros(len(inputs_pos)).long().to(device))

                    # total loss
                    loss = alpha * loss_cls + beta * loss_con + gamma * loss_loop
                else:
                    loss = loss_cls

                val_y_true.extend(labels.cpu().detach().numpy())
                val_y_scores.extend(Y_prob[:, 1].cpu().detach().numpy())
                val_y_pred.extend(Y_hat.cpu().detach().numpy())

                val_loss += loss.item() * inputs.size(0)

                if i % args.log_step == 0:
                    logging.info(
                        "Validating epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, val_loader_len) +
                        "loss_cls:{:.3f} ".format(loss_cls.item()) +
                        "loss_neg_patch_cls:{:.3f} ".format(loss_neg_patch_cls.item()) +
                        "loss_pos_patch_cls_unsup:{:.3f} ".format(loss_pos_patch_cls_unsup.item()) +
                        "loss_pos_patch_cls_sup:{:.3f} ".format(loss_pos_patch_cls_sup.item()) +
                        "loss_patch:{:.3f} ".format(loss_patch.item()) +
                        "loss_tm:{:.3f} ".format(loss_tm.item()) +
                        "loss_con:{:.3f} ".format(loss_con.item()) +
                        "loss_loop:{:.3f} ".format(loss_loop.item()) +
                        "loss:{:.3f}".format(loss.item())
                    )

        auc, precision, sensitivity, specificity, f1score = evaluate_model(y_true, y_scores, y_pred)
        val_auc, val_precision, val_sensitivity, val_specificity, val_f1score = evaluate_model(val_y_true, val_y_scores, val_y_pred)

        logging.info(
            "[Epoch:{:<5}/{:<5}] ".format(epoch+1, args.epochs) +
            "auc:{:.3f} precision:{:.3f}, sensitivity:{:.3f} specificity:{:.3f} f1_score:{:.3f} loss:{:.3f} ".format(
                auc, precision, sensitivity, specificity, f1score, running_loss / len(train_loader.dataset)) +
            "val_auc:{:.3f} val_precision:{:.3f}, val_sensitivity:{:.3f} val_specificity:{:.3f} val_f1_score:{:.3f} val_loss:{:.3f} ".format(
                val_auc, val_precision, val_sensitivity, val_specificity, val_f1score, val_loss / len(val_loader.dataset))
        )

        # Train
        train_epoch_auc.append(auc)
        train_epoch_precision.append(precision)
        train_epoch_sensitivity.append(sensitivity)
        train_epoch_specificity.append(specificity)
        train_epoch_f1score.append(f1score)
        train_epoch_loss.append(running_loss / len(train_loader.dataset))
        # Val
        val_epoch_auc.append(val_auc)
        val_epoch_precision.append(val_precision)
        val_epoch_sensitivity.append(val_sensitivity)
        val_epoch_specificity.append(val_specificity)
        val_epoch_f1score.append(val_f1score)
        val_epoch_loss.append(val_loss / len(val_loader.dataset))

        plot_history(train_epoch_auc, train_epoch_precision, train_epoch_sensitivity, train_epoch_specificity, train_epoch_f1score, train_epoch_loss,
                     val_epoch_auc, val_epoch_precision, val_epoch_sensitivity, val_epoch_specificity, val_epoch_f1score, val_epoch_loss, history_save_path)

        # save checkpoint
        model_save = copy.deepcopy(model)
        if best_val is None or val_auc > best_val:
            best_val = val_auc
            torch.save(model_save.state_dict(), os.path.join(ckpt_path, "best.pth"))
            logging.info("Saved best model.")
        torch.save(model_save.state_dict(), os.path.join(ckpt_path, "model_{}.pth".format(epoch + 1)))

    logging.info('STOP TIME:{}'.format(time.asctime(time.localtime(time.time()))))


def main():
    # init model
    model = CRCModel(model_name=args.model_name, p_cls=args.p_cls)
    model = nn.DataParallel(model)
    model.to(device)

    if args.p_cls and args.pretrain_weight_path:
        model.load_state_dict(torch.load(args.pretrain_weight_path))

    logging.info('START TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info(vars(args))

    data_crc_train = 'data/crc_train.txt'
    data_crc_val = 'data/crc_val.txt'

    crc_train_path, crc_val_path = [], []
    # Train
    for line in open(data_crc_train):
        crc_train_path.append(ast.literal_eval(line.strip()))
    # Test
    for line in open(data_crc_val):
        crc_val_path.append(ast.literal_eval(line.strip()))

    # data augmentation
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.GaussNoise(var_limit=0.01, p=0.5),
        A.OneOf([
            A.GridDistortion(p=0.5),
            A.ElasticTransform(p=0.5),
        ], p=0.75)
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))

    if args.p_cls:
        train_dataset = CRCPatchDataset(crc_train_path, transform=train_transform, type='train')
        val_dataset = CRCPatchDataset(crc_val_path, type='val')
    else:
        train_dataset = CRCDataset(crc_train_path, mode=args.mode, transform=train_transform, type='train')
        val_dataset = CRCDataset(crc_val_path, mode=args.mode, type='val')

    # dataset sampler
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=24, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True)

    logging.info("Train slices: {}, Val slices: {}".format(len(train_loader.dataset), len(val_loader.dataset)))

    # image classification loss
    criterion = nn.CrossEntropyLoss()

    # fuzzy classification loss
    criterion_fuzzy = FuzzyClassificationLoss()

    logging.info('Train Model')
    train(model, train_loader, val_loader, criterion, criterion_fuzzy)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CRC diagnosis")
    parser.add_argument("--epochs", type=int, default=10,
                        help="training epochs")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="initial learning rate")
    parser.add_argument("--log_step", type=int, default=200,
                        help="log accuracy each log_step batchs")
    parser.add_argument("--model_name", type=str, default="convnextv2_large.fcmae_ft_in22k_in1k",
                        help="model name")
    parser.add_argument("--mode", type=str, default='NAP',
                        help="mode to train")
    parser.add_argument("--p_cls", action="store_true", default=False,
                        help="patch cls")
    parser.add_argument("--pretrain_weight_path", type=str, default="",
                        help="pretrain weight path")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="experiment name")

    args = parser.parse_args()

    ckpt_path = os.path.join('run/ckpt', args.exp_name)
    logs_path = os.path.join('run/logs', args.exp_name, "{}-{}.log".format(datetime.now(), args.exp_name)).replace(":", ".")
    history_save_path = os.path.join('run/history', args.exp_name, "{}-{}.png".format(datetime.now(), args.exp_name)).replace(":", ".")

    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(os.path.join('run/logs', args.exp_name), exist_ok=True)
    os.makedirs(os.path.join('run/history', args.exp_name), exist_ok=True)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(logs_path, mode='a'), logging.StreamHandler()]
    )

    try:
        main()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        exit(1)
