import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class FaultDetectionMaskRCNN(nn.Module):
    def __init__(self, num_classes=5):  # 4种故障 + 1个背景
        super(FaultDetectionMaskRCNN, self).__init__()

        # 加载预训练的Mask R-CNN模型，使用新的weights参数
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

        # 获取分类器的输入特征数
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # 替换分类头
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # 获取mask预测器的输入特征数
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # 替换mask预测头
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

def get_model(num_classes=5):
    """获取配置好的Mask R-CNN模型"""
    return FaultDetectionMaskRCNN(num_classes)