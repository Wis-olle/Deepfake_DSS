"""Lớp trừu tượng cho mọi detector — chép nguyên từ DeepfakeBench/training/detectors/base_detector.py (Zhiyuan Yan, 2023).
Hợp đồng: mọi detector nhận data_dict {'image','label',...} và trả pred_dict {'cls','prob','feat'};
trainer chỉ gọi forward / get_losses / get_train_metrics, không cần biết bên trong là CNN gì."""
import abc
import torch
import torch.nn as nn


class AbstractDetector(nn.Module, metaclass=abc.ABCMeta):
    """All deepfake detectors should subclass this class."""
    def __init__(self, config=None, load_param=False):
        super().__init__()

    @abc.abstractmethod
    def features(self, data_dict: dict) -> torch.Tensor:
        """Đặc trưng từ backbone."""

    @abc.abstractmethod
    def forward(self, data_dict: dict, inference=False) -> dict:
        """Lan truyền xuôi, trả pred_dict."""

    @abc.abstractmethod
    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """Từ đặc trưng ra logit các lớp."""

    @abc.abstractmethod
    def build_backbone(self, config):
        """Dựng backbone."""

    @abc.abstractmethod
    def build_loss(self, config):
        """Dựng hàm mất mát."""

    @abc.abstractmethod
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """Tính các mất mát; khoá 'overall' là cái được lan truyền ngược."""

    @abc.abstractmethod
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        """Chỉ số trên lô huấn luyện."""
