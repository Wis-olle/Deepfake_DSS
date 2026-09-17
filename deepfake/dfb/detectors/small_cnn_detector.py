"""SmallCNN — mạng tích chập nhỏ tự dựng (97 nghìn tham số), viết theo khuôn resnet34_detector.py của DeepfakeBench.
Chín phương thức y hệt họ: __init__, build_backbone, build_loss, features, classifier, get_losses, get_train_metrics, forward.
Khác họ: đầu ra 2 lớp + softmax (họ cũng vậy) nhưng lớp cuối khởi tạo 0 → loss khởi tạo đúng ln2 (bài học từ sổ tay, Bước 4)."""
import torch
import torch.nn as nn

from metrics.utils import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR, LOSSFUNC


def block(cin, cout):   # một "tầng": tích chập 3×3 → chuẩn hoá lô → ReLU → gộp cực đại 2×2
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))


@DETECTOR.register_module(module_name="small_cnn")
class SmallCNNDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(config["backbone_config"]["width"][-1], 2)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)   # logit hai lớp bằng nhau → loss = ln2
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        layers, cin = [], 3
        for cout in config["backbone_config"]["width"]:
            layers.append(block(cin, cout)); cin = cout
        return nn.Sequential(*layers)                       # 96→48→24→12→6 với 4 tầng

    def build_loss(self, config):
        return LOSSFUNC[config["loss_func"]]()

    def features(self, data_dict):
        return self.backbone(data_dict["image"]).mean(dim=(2, 3))   # gộp trung bình toàn cục

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
        prob = torch.softmax(pred, dim=1)[:, 1]              # xác suất "giả"
        return {"cls": pred, "prob": prob, "feat": features}
