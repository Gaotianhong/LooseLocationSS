import os
import cv2
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, accuracy_score

from models.config import IMG_SIZE, GRID_SIZE


def set_seed(seed):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_idxs(n_selection, start, end):
    """Generate random indexes"""
    x = np.arange(start, end, 1)
    replace = False if (end - start) >= n_selection else True
    res = np.random.choice(x, size=n_selection, replace=replace)
    res = sorted(res)
    return res


def evaluate_model(y_true, y_scores, y_pred):
    """Evaluate model"""
    auc = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1score = f1_score(y_true, y_pred, zero_division=0)

    return auc, precision, sensitivity, specificity, f1score


def test_model_on_dataset(model, data_loader, device, cls_threshold):
    """Slice level performance"""
    total = len(data_loader)
    y_true, y_scores, y_pred = [], [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            print("{:<5}/{:<5}".format(i+1, total), end="\r")
            inputs, volumes, boxes, labels = data
            inputs = inputs.to(device)
            volumes = volumes.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            outputs = model(inputs, volumes)
            Y_prob = F.softmax(outputs, dim=1)

            Y_hat = (Y_prob[:, 1].cpu().detach().numpy() >= cls_threshold).astype(int)
            y_pred.extend(Y_hat)

            y_true.extend(labels.cpu().detach().numpy())
            y_scores.extend(Y_prob[:, 1].cpu().detach().numpy())

        print("{:<5}/{:<5}".format(i+1, total))

    acc = accuracy_score(y_true, y_pred)
    auc, precision, sensitivity, specificity, f1score = evaluate_model(y_true, y_scores, y_pred)
    print("acc:{:.2f} auc:{:.2f} precision:{:.2f} sensitivity:{:.2f} specificity:{:.2f} f1_score:{:.2f} ".format(
        acc*100, auc*100, precision*100, sensitivity*100, specificity*100, f1score*100))

    return acc, auc, precision, sensitivity, specificity, f1score


def test_model_demo(model, data_loader, threshold, cls_threshold, crc_test_path, demo_model_name, post, device):
    """Patient-level performance"""
    if crc_test_path is not None:
        mode = crc_test_path[0].split('/')[-2]
        cls_post = str(cls_threshold).replace('.', '')
        log_path = f'run/model_test/{demo_model_name}_{cls_post}/{mode}pred_{threshold}_{post}.log'
        if os.path.exists(log_path):
            os.remove(log_path)

    total = len(data_loader)
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            print("{:<5}/{:<5}".format(i+1, total), end="\r")
            inputs, volumes, labels, label_nums = data
            inputs = inputs.to(device)
            volumes = volumes.to(device)
            labels = labels.to(device)
            label_nums = label_nums.to(device)

            outputs = model(inputs, volumes)
            Y_prob = F.softmax(outputs, dim=1)
            Y_hat = (Y_prob[:, 1].cpu().detach().numpy() >= cls_threshold).astype(int)
            y_pred_all = np.array(Y_hat)

            y_pred_label = 1 if np.sum(y_pred_all) > threshold else 0

            y_true.extend(labels.cpu().detach().numpy())  # true label
            y_pred.append(y_pred_label)  # predict

        print("{:<5}/{:<5}".format(i+1, total))

    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1score = f1_score(y_true, y_pred)
    print("acc:{:.2f} precision:{:.2f} sensitivity:{:.2f} specificity:{:.2f} f1_score:{:.2f} ".format(
        acc*100, precision*100, sensitivity*100, specificity*100, f1score*100))

    result = [acc, precision, sensitivity, specificity, f1score]

    return result


def plot_history(train_epoch_auc, train_epoch_precision, train_epoch_sensitivity, train_epoch_specificity, train_epoch_f1score, train_epoch_loss,
                 val_epoch_auc, val_epoch_precision, val_epoch_sensitivity, val_epoch_specificity, val_epoch_f1score, val_epoch_loss, save_path):
    """Plot training history"""
    _, ax = plt.subplots(2, 3, figsize=(12, 18))

    # AUC
    ax[0, 0].plot(train_epoch_auc)
    ax[0, 0].plot(val_epoch_auc)
    ax[0, 0].set_title("Model {}".format("AUC"))
    ax[0, 0].set_xlabel("epochs")
    ax[0, 0].legend(["train", "val"])

    # Loss
    ax[0, 1].plot(train_epoch_loss)
    ax[0, 1].plot(val_epoch_loss)
    ax[0, 1].set_title("Model {}".format("Loss"))
    ax[0, 1].set_xlabel("epochs")
    ax[0, 1].legend(["train", "val"])

    # Precision
    ax[0, 2].plot(train_epoch_precision)
    ax[0, 2].plot(val_epoch_precision)
    ax[0, 2].set_title("Model {}".format("Precision"))
    ax[0, 2].set_xlabel("epochs")
    ax[0, 2].legend(["train", "val"])

    # Sensitivity
    ax[1, 0].plot(train_epoch_sensitivity)
    ax[1, 0].plot(val_epoch_sensitivity)
    ax[1, 0].set_title("Model {}".format("Sensitivity"))
    ax[1, 0].set_xlabel("epochs")
    ax[1, 0].legend(["train", "val"])

    # Specificity
    ax[1, 1].plot(train_epoch_specificity)
    ax[1, 1].plot(val_epoch_specificity)
    ax[1, 1].set_title("Model {}".format("Specificity"))
    ax[1, 1].set_xlabel("epochs")
    ax[1, 1].legend(["train", "val"])

    # F1 Score
    ax[1, 2].plot(train_epoch_f1score)
    ax[1, 2].plot(val_epoch_f1score)
    ax[1, 2].set_title("Model {}".format("F1 Score"))
    ax[1, 2].set_xlabel("epochs")
    ax[1, 2].legend(["train", "val"])

    plt.savefig(save_path)
    plt.close()


def plot_slices(num_rows, num_columns, width, height, scan_mode, data, data_dir=None):
    """Plot CT slices"""
    data = np.transpose(data, (2, 0, 1))
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 6.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            if num_rows == 1:
                axarr[j].text(100, 50, 1 + j + i * columns_data, color='white', fontsize=5)
                axarr[j].imshow(data[i][j], cmap="gray")
                axarr[j].axis("off")
            else:
                axarr[i, j].text(100, 50, 1 + j + i * columns_data, color='white', fontsize=5)
                axarr[i, j].imshow(data[i][j], cmap="gray")
                axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    if data_dir == None:
        plt.savefig("figure/{}CT.png".format(scan_mode), dpi=500)
    else:
        figure_path = os.path.join('figure/{}'.format(data_dir))
        if not os.path.exists(figure_path):
            os.makedirs(figure_path, exist_ok=True)
        plt.savefig("figure/{}/{}CT.png".format(data_dir, scan_mode), dpi=500)


def get_bounding_boxes(heatmap, threshold=0.15, otsu=False):
    """Get bounding boxes from heatmap"""
    p_heatmap = np.array(heatmap*255, np.uint8)
    if otsu:
        # Otsu's thresholding method to find the bounding boxes
        threshold, p_heatmap = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Using a fixed threshold
        p_heatmap[p_heatmap < threshold * 255] = 0
        p_heatmap[p_heatmap >= threshold * 255] = 1

    # find the contours in the thresholded heatmap
    contours = cv2.findContours(p_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # get the bounding boxes from the contours
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append([x, y, x + w, y + h])

    return bboxes


def get_circle_patches(heatmap, radius=3, color='lightblue'):
    """Get patches for circle point"""
    patches = []
    x = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[0]
    y = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[1]

    patches.append(Circle((y, x), radius=radius, color=color))
    return patches


def get_bbox_patches(bboxes, color='r', linewidth=2):
    """Get patches for bounding boxes"""
    patches = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        patches.append(Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', linewidth=linewidth))

    return patches


def create_animation(array, case, box=None, pred_box=None, heatmap=None, alpha=0.3, animation_path=None):
    """Create an animation of images"""
    fig = plt.figure(figsize=(4, 4))
    images = []

    grid_size = GRID_SIZE
    step_size = IMG_SIZE // grid_size

    for idx, image in enumerate(array):
        # plot gridlines
        for x in range(0, image.shape[1], step_size):
            image[:, x:x+1] = 1
        for y in range(0, image.shape[0], step_size):
            image[y:y+1, :] = 1

        image_plot = plt.imshow(image, animated=True, cmap='gray')
        aux = [image_plot]
        if box is not None:
            # add groundtruth bounding boxes to the heatmap image as animated patches
            patches = get_bbox_patches([box[0][idx]], color='red')
            aux.extend(image_plot.axes.add_patch(patch) for patch in patches)

        if pred_box is not None:
            # add pred bounding boxes to the heatmap image as animated patches
            patches = get_bbox_patches([pred_box[idx][0]], color='blue')  # fix
            aux.extend(image_plot.axes.add_patch(patch) for patch in patches)

        if heatmap is not None:
            image_plot2 = plt.imshow(heatmap[idx], animated=True, cmap='jet', alpha=alpha, extent=image_plot.get_extent())
            aux.append(image_plot2)
            # add maximally activated point
            patches = get_circle_patches(heatmap[idx])
            # aux.extend(image_plot2.axes.add_patch(patch) for patch in patches)

            # add bounding boxes to the heatmap image as animated patches
            bboxes = get_bounding_boxes(heatmap[idx])
            patches = get_bbox_patches(bboxes, color='blue')
            # aux.extend(image_plot2.axes.add_patch(patch) for patch in patches)

        images.append(aux)

    plt.axis('off')
    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, images, interval=3000//len(array), blit=False, repeat_delay=1500)
    plt.close()

    if not os.path.exists(animation_path):
        os.makedirs(animation_path, exist_ok=True)

    ani.save(os.path.join(animation_path, 'animation_{}.gif'.format(case)))

    return ani


def get_region(input):
    """Get highest lesion location"""
    class1_probabilities = input[:, 1, :, :]
    _, max_index = torch.max(class1_probabilities.view(-1), 0)
    max_position = (max_index // GRID_SIZE, max_index % GRID_SIZE)
    # print(f"Maximum probability for class 1 is at position: {max_position}")
    y, x = max_position[0].item(), max_position[1].item()

    return y, x


def generate_mask(input_tensor):
    """Generate lesion mask"""
    batch_size, _, height, width = input_tensor.shape
    mask_tensor = torch.ones_like(input_tensor)

    for b in range(batch_size):
        class_1_probs = input_tensor[b, 1, :, :]

        _, max_index = torch.max(class_1_probs.view(-1), 0)
        max_y, max_x = max_index // GRID_SIZE, max_index % GRID_SIZE
        y1 = max(max_y-1, 0)
        y2 = min(max_y+2, height)
        x1 = max(max_x-1, 0)
        x2 = min(max_x+2, width)
        mask_tensor[b, :, y1:y2, x1:x2] = 0

    return mask_tensor
