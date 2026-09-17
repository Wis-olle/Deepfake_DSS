"""ResNet-18 tiền huấn luyện ImageNet — theo khuôn resnet34_detector.py của DeepfakeBench
(họ dùng ResNet-34 cho ngang số tham số với Xception; ta dùng ResNet-18 vì chạy CPU).
Trọng số ImageNet nạp ngay trong build_backbone (họ ghi FIXME vì nạp ở nơi khác)."""
import torch
import torch.nn as nn
import torchvision

from metrics.utils import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR, LOSSFUNC


@DETECTOR.register_module(module_name="resnet18")
class ResNet18Detector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(512, 2)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if config["backbone_config"].get("pretrained", True) else None
        m = torchvision.models.resnet18(weights=w)
        m.fc = nn.Identity()                                  # bỏ lớp 1000 lớp ImageNet, giữ vector 512
        return m

    def build_loss(self, config):
        return LOSSFUNC[config["loss_func"]]()

    def features(self, data_dict):
        return self.backbone(data_dict["image"])

    def classifier(self, features):
        return self.head(features)

    def get_losses(self, data_dict, pred_dict):
        return {"overall": self.loss_func(pred_dict["cls"], data_dict["label"])}

    def get_train_metrics(self, data_dict, pred_dict):
        auc, eer, acc, ap = calculate_metrics_for_train(data_dict["label"].detach(), pred_dict["cls"].detach())
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}

    def forward(self, data_dict, inference=False):
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {"cls": pred, "prob": prob, "feat": features}
