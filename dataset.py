import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class FaultDetectionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms

        # 按图片分组标签
        self.grouped = self.df.groupby('ImageId')
        self.img_ids = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.img_ids)

    def decode_rle(self, rle_string, height, width):
        """解码RLE编码的像素"""
        if pd.isna(rle_string):
            return np.zeros((height, width), dtype=np.uint8)

        rle = list(map(int, rle_string.split()))
        starts = rle[0::2]
        lengths = rle[1::2]

        mask = np.zeros(height * width, dtype=np.uint8)
        for start, length in zip(starts, lengths):
            mask[start - 1:start - 1 + length] = 1

        return mask.reshape((width, height)).T

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # 加载图片
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # 获取该图片的所有标注
        img_annotations = self.grouped.get_group(img_id)

        boxes = []
        labels = []
        masks = []

        height, width = image.shape[:2]

        for _, row in img_annotations.iterrows():
            class_id = int(row['ClassId'])
            encoded_pixels = row['EncodedPixels']

            # 解码mask
            mask = self.decode_rle(encoded_pixels, height, width)

            if mask.sum() > 0:  # 确保mask不为空
                # 从mask计算边界框
                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)
                masks.append(mask)

        # 转换为tensor
        if len(boxes) == 0:
            # 如果没有标注，创建空的tensor
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)

        return image, target


def get_transform(train=True):
    """获取数据预处理变换"""
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])