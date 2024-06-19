import os
import ast
import torch
import argparse
import numpy as np
import torch.nn as nn

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import CRCDemoDataset
from models.crc_model import CRCModel
from models.config import IMG_SIZE, GRID_SIZE, MODE_ORDER
from utils.utils import set_seed, create_animation, get_region, get_bounding_boxes


set_seed(seed=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=1,
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
    )
    parser.add_argument(
        "--p_cls", action="store_true", default=False, help="patch cls"
    )
    parser.add_argument(
        "--vis", action="store_true", default=False, help="visualization"
    )
    parser.add_argument(
        "--pretrain_weight_path", type=str, default="", help="pretrain weight path"
    )

    return parser.parse_args()


def get_grid_matrix(box):
    """Divide image into grid matrix"""
    image_size = (IMG_SIZE, IMG_SIZE)
    grid_size = (GRID_SIZE, GRID_SIZE)
    cell_size = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

    matrix = np.zeros(grid_size)

    xmin, ymin, xmax, ymax = box

    grid_xmin = int(xmin // cell_size[0])
    grid_ymin = int(ymin // cell_size[1])
    grid_xmax = int(xmax // cell_size[0])
    grid_ymax = int(ymax // cell_size[1])

    for x in range(grid_xmin, grid_xmax + 1):
        for y in range(grid_ymin, grid_ymax + 1):
            matrix[y, x] = 1

    return matrix


def eval_pcls():
    """Evaluate lesion localization for patch cls"""
    TP = 0
    FP = 0
    FN = 0
    cell_size = IMG_SIZE // GRID_SIZE

    for i, data in enumerate(test_loader, 0):
        print("{:<5}/{:<5}".format(i+1, total), end="\r")
        inputs, _, boxes, labels, _ = data
        labels = labels.to(device)
        for j in range(len(inputs)):  # N mode eval
            if j > 0:
                break
            image, pred_boxes = [], []
            for k in range(len(inputs[j][0])):
                input_tensor = inputs[j][0][k].unsqueeze(0).to(device)  # image
                image.append(np.squeeze(input_tensor.cpu().detach().numpy()))

                _, outputs_p_cls = model(input_tensor)

                max_y, max_x = get_region(outputs_p_cls)
                nei0 = 0.5
                nei1 = 2 - nei0
                pred_boxes.append(np.array([[(max_y+nei0)*cell_size, (max_x+nei0)*cell_size, min((max_y+nei1)*cell_size, IMG_SIZE-1), min((max_x+nei1)*cell_size, IMG_SIZE-1)]]))

            for pred_box, true_box in zip(pred_boxes,  boxes[j][0].cpu().detach().numpy()):
                # get grid matrix
                pred_grid = get_grid_matrix(pred_box[0])
                gt_grid = get_grid_matrix(true_box)

                TP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 1))
                FP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 0))
                FN += np.sum(np.logical_and(pred_grid == 0, gt_grid == 1))

            if args.vis:
                create_animation(image, f'{i+1}{MODE_ORDER[j]}mt', boxes[j].cpu().detach().numpy(),
                                 pred_boxes, None, animation_path=animation_path)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    iou = TP / (TP + FP + FN)
    print('precision:{:.2f} recall:{:.2f} iou:{:.2f}'.format(precision*100, recall*100, iou*100))


def eval_cam():
    """eval cam"""
    target_layers = [model.module.model.stages[3].blocks[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    # cam = EigenCAM(model=model, target_layers=target_layers)

    TP = 0
    FP = 0
    FN = 0

    for i, data in enumerate(test_loader, 0):
        print("{:<5}/{:<5}".format(i+1, total), end="\r")
        inputs, _, boxes, labels, input_masks = data
        labels = labels.to(device)
        images, heatmaps, masks = [], [], []
        y_trues, y_preds = [], []
        for j in range(len(inputs)):
            if j > 0:  # only for N
                break
            image, heatmap, mask = [], [], []
            y_true, y_pred = [], []
            for k in range(len(inputs[j][0])):
                input_tensor = inputs[j][0][k].unsqueeze(0).to(device)
                # GradCAM or EigenCAM
                targets = [ClassifierOutputTarget(labels)]
                grayscale_cam = cam(input_tensor, targets)
                grayscale_cam = grayscale_cam[0, :]
                image.append(np.squeeze(input_tensor.cpu().detach().numpy()))

                grayscale_cam[grayscale_cam <= args.threshold] = 0  # shape:(224, 224)

                y_true.append(labels.cpu().detach().numpy()[0])
                y_pred.append(torch.argmax(cam.outputs, dim=1).cpu().detach().numpy()[0])

                heatmap.append(grayscale_cam)
            for input_mask in input_masks[j]:  # mask
                mask.append(np.squeeze(input_mask.cpu().detach().numpy()))

            images.append(image)
            masks.append(mask)
            heatmaps.append(heatmap)
            y_trues.append(y_true)
            y_preds.append(y_pred)

            if args.vis:
                create_animation(image, f'{i+1}{MODE_ORDER[j]}heatmap', box=boxes[j].cpu().detach().numpy(), pred_box=None,
                                 heatmap=heatmap, animation_path=animation_path)

        # evaluate cam
        for j in range(len(heatmaps)):
            for k in range(len(heatmaps[j])):
                pred_box = get_bounding_boxes(heatmaps[j][k])
                true_box = get_bounding_boxes(masks[j][k])

                pred_grid = get_grid_matrix(pred_box[0])
                gt_grid = get_grid_matrix(true_box[0])

                TP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 1))
                FP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 0))
                FN += np.sum(np.logical_and(pred_grid == 0, gt_grid == 1))

    print("{:<5}/{:<5}\n".format(i+1, total))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    iou = TP / (TP + FP + FN)
    print('precision:{:.2f} recall:{:.2f} iou:{:.2f}'.format(precision*100, recall*100, iou*100))


if __name__ == '__main__':
    args = parse_arguments()

    crc_path = []
    for line in open('data/crc_val.txt'):
        temp_path = ast.literal_eval(line.strip())
        if temp_path[0].find('abnormal') != -1:
            crc_path.append(ast.literal_eval(line.strip()))  # abnormal

    test_dataset = CRCDemoDataset(crc_path, mode='NAP', test_cam=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True)
    total = len(test_loader)
    print("Total patients: {}".format(len(test_loader.dataset)))

    # load pretrained model
    model = CRCModel(p_cls=args.p_cls, test_cam=True)
    model = nn.DataParallel(model)
    model_name = args.pretrain_weight_path.split('/')[-2]
    model.load_state_dict(torch.load(args.pretrain_weight_path))

    # evaluate
    model.to(device)
    model.eval()

    vis = 'vis' if args.vis else ''
    if args.p_cls:
        animation_path = os.path.join('run', 'vis', f'{model_name}_mt_{vis}')
        eval_pcls()
    else:
        animation_path = os.path.join('run', 'vis', f'{model_name}_cam_{vis}')
        eval_cam()
