"""Sổ đăng ký detector — học theo DeepfakeBench/training/metrics/registry.py và detectors/__init__.py.
Mỗi tệp *_detector.py tự đăng ký lớp của nó bằng @DETECTOR.register_module(module_name=...);
train.py chỉ cần DETECTOR[config["model_name"]] — thêm detector mới = thêm một tệp, không sửa train.py."""
import torch.nn as nn


class Registry:
    def __init__(self): self.data = {}
    def register_module(self, module_name=None):
        def _register(cls):
            self.data[module_name or cls.__name__] = cls; return cls
        return _register
    def __getitem__(self, key): return self.data[key]
    def keys(self): return list(self.data)


DETECTOR = Registry()
LOSSFUNC = {"cross_entropy": nn.CrossEntropyLoss}     # DeepfakeBench để ở loss/; ở đây một dict là đủ

from . import small_cnn_detector, resnet18_detector, sbi_detector   # noqa: E402  (nạp để các lớp tự đăng ký)
