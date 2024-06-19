import os
import ast
import cv2
import json
import random
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset

from utils.utils import random_idxs
from models.config import IMG_SIZE, MODE_ORDER, CRC_MATCH_PATH, CRC_INDEXES_PATH, CRC_ALL_PATH


class CRCDataset(Dataset):
    """CRC scan dataset"""

    def __init__(self, crc_path, mode, transform=None, type='train'):

        super().__init__()

        self.imgs, self.boxes, self.labels = [], [], [], []
        self.transform = transform

        mode_index = []
        for m in mode:
            mode_index.append(MODE_ORDER.index(m))

        # lesion annotations path
        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))

        # slice index
        indexes_dict = {}
        if os.path.exists(CRC_INDEXES_PATH):
            indexes_dict = json.load(open(CRC_INDEXES_PATH, 'r'))

        # random sampling lesion and healthy sequence
        var_length_abnormal = np.arange(3, 6)
        var_length_normal = np.arange(10, 21)

        crc_path = list(map(list, zip(*crc_path)))
        abnormal_num, abnormal_slices, normal_num, normal_slices = 0, 0, 0, 0
        for i in mode_index:
            scans, boxes, labels = [], [], [], []
            for path in crc_path[i]:
                scan = np.load(path)
                num_slice = scan.shape[2]
                if path in indexes_dict:
                    # load indexes from json
                    indexes = ast.literal_eval(indexes_dict[path])
                else:
                    # 'start' and 'end' denote colon sequence
                    start, end = 25, num_slice - 12

                    if path.find('abnormal') != -1:
                        num_scans = np.random.choice(var_length_abnormal)
                    else:
                        num_scans = np.random.choice(var_length_normal)
                    normal_indexes = random_idxs(num_scans, start, end)  # random

                    abnormal_indexes = []
                    if path in match_dict:  # abnormal
                        for line in open(os.path.join(match_dict[path], 'slice.txt')):
                            abnormal_indexes.append(int(line.split('\t')[0]) - 1)

                    indexes = sorted(list(set(abnormal_indexes + normal_indexes)))
                    indexes_dict[path] = str(indexes)
                label = np.zeros(num_slice)
                # abnormal slice
                slice_box = {}
                if path in match_dict:
                    for line in open(os.path.join(match_dict[path], 'slice.txt')):
                        slice, bbox = int(line.split('\t')[0]), ast.literal_eval(line.split('\t')[3].replace('\n', ''))
                        label[slice - 1] = 1
                        slice_box[slice - 1] = bbox  # bounding box

                # scan
                scan = np.array([scan[:, :, idx] for idx in indexes])
                box = []
                for idx in indexes:
                    if idx in slice_box:
                        temp_slice_box = [coor/512 for coor in slice_box[idx]]
                        box.append(temp_slice_box)  # abnormal
                    else:
                        box.append([0, 0, 1.0, 1.0])  # normal
                label = np.array([label[idx] for idx in indexes])
                scans.append(scan)
                boxes.append(box)
                labels.append(label)

                if path.find('abnormal') != -1:
                    abnormal_num += 1
                else:
                    normal_num += 1
                abnormal_slices += len(label[label == 1])
                normal_slices += len(label[label == 0])

            # merge N A P modality
            for j in range(len(scans)):
                self.imgs.extend(img for img in scans[j])
                self.boxes.extend(box for box in boxes[j])
                self.labels.extend(label for label in labels[j])

        with open(CRC_INDEXES_PATH, 'w') as f:
            json.dump(indexes_dict, f, ensure_ascii=False, indent=4)

        self.num_per_cls_dict = [normal_slices, abnormal_slices]
        print('{}, abnormal patients={} slices={}, normal patients={} slices={}'.format(
            type, int(abnormal_num/len(mode_index)), abnormal_slices, int(normal_num/len(mode_index)), normal_slices))

    def __getitem__(self, idx):
        img = self.imgs[idx]
        volume = -1
        box = self.boxes[idx]
        label = self.labels[idx]

        if self.transform:
            transformed = self.transform(image=img, bboxes=[box], class_labels=[label])
            img = transformed['image']
            box = transformed['bboxes']

        img = np.expand_dims(img, axis=0)
        volume = np.expand_dims(volume, axis=0)
        if label == 0:
            box = [(-1.0, -1.0, -1.0, -1.0)]
        if len(box) == 1:  # unify train and test
            box = box[0]
        box = np.array(box).astype('float')
        label = np.array(label).astype('int')

        return img, volume, box, label

    def __len__(self):
        return len(self.labels)


class CRCPatchDataset(Dataset):
    """CRC scan dataset for p_cls"""

    def __init__(self, crc_path, transform=None, type='train'):

        super().__init__()

        self.paths, self.paths3D, self.labels = [], [], []
        self.transform = transform

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))
        indexes_dict = json.load(open(CRC_INDEXES_PATH, 'r'))
        crc_all_dict = json.load(open(CRC_ALL_PATH, 'r'))

        abnormal_num, box_num, normal_num = 0, 0, 0
        for i in range(len(crc_path)):
            pathN, pathA, pathP = crc_path[i]
            num_slice = int(crc_all_dict[pathN])
            label = np.zeros(num_slice)

            slice_box = {}
            idxN, idxA, idxP = [], [], []
            if pathN in match_dict:
                abnormal_num += 1
                for line in open(os.path.join(match_dict[pathN], 'slice.txt')):
                    slice, gt, bbox = int(line.split('\t')[0]), int(line.split('\t')[2]), ast.literal_eval(line.split('\t')[3].replace('\n', ''))
                    idxN.append(slice - 1)
                    label[slice - 1] = 1  # abnormal label
                    slice_box[slice - 1] = bbox  # bounding box
                    if gt == 1:
                        box_num += 1
                for line in open(os.path.join(match_dict[pathA], 'slice.txt')):
                    idxA.append(int(line.split('\t')[0]) - 1)
                for line in open(os.path.join(match_dict[pathP], 'slice.txt')):
                    idxP.append(int(line.split('\t')[0]) - 1)
                assert len(slice_box) == len(idxA) == len(idxP)
            else:
                normal_num += 1

            k = 0
            indexes = ast.literal_eval(indexes_dict[pathN])
            for idx in indexes:
                temp_slice_box = [0, 0, 1.0, 1.0]  # normal
                temp_path3D = [(pathA, idx), idx-1, idx+1, (pathP, idx)]  # normal
                if idx in slice_box:  # abnormal
                    temp_slice_box = [coor/512 for coor in slice_box[idx]]
                    high = idx + 1 if idx + 1 < num_slice else idx
                    temp_path3D = [(pathA, idxA[k]), idx-1, high, (pathP, idxP[k])]
                    k += 1
                self.paths.append((pathN, idx, temp_slice_box))
                self.paths3D.append(temp_path3D)
                self.labels.append(label[idx])
                # print(self.paths[-1], self.paths3D[-1], self.labels[-1])

        self.labels = np.array(self.labels)
        print('{}, abnormal patients={} slices={}, boxes={}, normal patients={} slices={}'.format(
            type, abnormal_num, np.sum(self.labels == 1), box_num, normal_num, np.sum(self.labels == 0)))

    def __getitem__(self, idx):
        # npy path
        path = self.paths[idx]
        path3D = self.paths3D[idx]

        # mode N
        scanN, scanA, scanP = np.load(path[0]), np.load(path3D[0][0]), np.load(path3D[-1][0])
        img = scanN[:, :, path[1]]
        volume = np.stack((scanA[:, :, path3D[0][1]], scanN[:, :, path3D[1]], scanN[:, :, path3D[2]], scanP[:, :, path3D[-1][1]]), axis=-1)

        box = path[2]
        label = self.labels[idx]

        if self.transform:
            transformed = self.transform(image=img, bboxes=[box], class_labels=[label])
            img = transformed['image']
            box = transformed['bboxes']
            # 3d data augmentation
            angles = [-60, -30, -15, 0, 15, 30, 60]
            angle = random.choice(angles)
            volume = ndimage.rotate(volume, angle, reshape=False)  # rotate CT
            volume[volume < 0] = 0
            volume[volume > 1] = 1

        img = np.expand_dims(img, axis=0)
        volume = np.expand_dims(volume, axis=0)
        if label == 0:
            box = [(-1.0, -1.0, -1.0, -1.0)]
        if len(box) == 1:  # unify train and test
            box = box[0]
        box = np.array(box).astype('float')
        label = np.array(label).astype('int')

        return img, volume, box, label

    def __len__(self):
        return len(self.labels)


class CRCDemoDataset(Dataset):
    """CRC patient-level dataset"""

    def __init__(self, crc_path, mode='N', test_cam=False):

        super().__init__()

        self.imgs, self.boxes, self.labels, self.label_nums = [], [], [], [], []

        self.mode_index = []
        for m in mode:
            self.mode_index.append(MODE_ORDER.index(m))
        self.test_cam = test_cam

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))
        indexes_dict = json.load(open(CRC_INDEXES_PATH, 'r'))

        # load scan
        for i in range(len(crc_path)):
            scans, boxes, label_nums = [], [], [], [], []
            label = 1 if crc_path[i][0].find('abnormal') != -1 else 0
            for m_idx in self.mode_index:  # NAP
                path = crc_path[i][m_idx]
                scan = np.load(path)
                num_slice = scan.shape[2]
                box, label_num = [], []
                if test_cam:
                    if path in match_dict:
                        indexes = []
                        for line in open(os.path.join(match_dict[path], 'slice.txt')):
                            slice, bbox = line.split('\t')[0], ast.literal_eval(line.split('\t')[1])
                            indexes.append(int(slice) - 1)
                            box.append(bbox)
                        sorted_lists = sorted(zip(indexes, box), key=lambda pair: pair[0])
                        indexes = [x[0] for x in sorted_lists]
                        box = [x[1] for x in sorted_lists]
                    else:
                        # normal
                        indexes = ast.literal_eval(indexes_dict[path])
                        box = []
                        for _ in range(len(indexes)):
                            box.append([0, 0, 0, 0])
                else:
                    # total sequence of patient, 'start' and 'end' denote colon sequence
                    start, end = 25, num_slice - 12
                    indexes = np.arange(start, end, 1)
                    if path in match_dict:
                        for line in open(os.path.join(match_dict[path], 'slice.txt')):
                            slice = line.split('\t')[0]
                            label_num.append(int(slice) - 1 - start)
                    else:
                        label_num.append(0)  # normal
                    label_nums.append(label_num)

                scan = [scan[:, :, idx] for idx in indexes]
                scans.append(scan)
                boxes.append(box)

            # merge N A P
            if len(self.mode_index) == 3 and not test_cam:
                scans = np.concatenate((scans[0], scans[1], scans[2]), axis=0)
                label_nums = np.concatenate((label_nums[0], label_nums[1], label_nums[2]), axis=0)

            self.imgs.append(scans)
            self.boxes.append(boxes)
            self.labels.append(label)
            self.label_nums.append(label_nums)

        self.labels = np.array(self.labels)
        print('abnormal: {}, normal: {}'.format(np.sum(self.labels == 1), np.sum(self.labels == 0)))

    def __getitem__(self, idx):
        img = self.imgs[idx][0] if len(self.mode_index) == 1 else self.imgs[idx]
        volume = -1
        label = self.labels[idx]

        if self.test_cam:
            box = self.boxes[idx]
            mask = []
            for i in range(len(img)):  # NAP
                img[i] = np.expand_dims(img[i], axis=1)
                if box[i][0] is not None:
                    temp_mask = []
                    for bbox in box[i]:
                        _mask = np.zeros((512, 512))
                        _mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = 1
                        _mask = np.array(cv2.resize(_mask, (IMG_SIZE, IMG_SIZE)))
                        temp_mask.append(_mask)
                    mask.append(temp_mask)  # bounding box generates mask
                    box[i] = np.array(box[i])*(IMG_SIZE/512)

            return img, volume, box, label, mask
        else:
            label_num = self.label_nums[idx][0]
            img = np.expand_dims(img, axis=1)
            volume = np.expand_dims(volume, axis=0)  # unify
            label_num = np.array(label_num)

            return img, volume, label, label_num

    def __len__(self):
        return len(self.labels)
