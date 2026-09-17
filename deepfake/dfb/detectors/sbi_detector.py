"""SBI (Shiohara & Yamasaki, CVPR 2022) — theo sbi_detector.py của DeepfakeBench.
Điểm mấu chốt: mã mô hình của SBI KHÔNG khác một bộ phân lớp thường (họ: EfficientNet-B4 + cross-entropy; ta: ResNet-18).
Toàn bộ "phép màu" nằm ở DỮ LIỆU: dataset/sbi_api.py tự tạo ảnh giả từ ảnh thật, dataset/sbi_dataset.py trả về cặp.
Vì vậy ở đây chỉ kế thừa ResNet18Detector và đổi tên đăng ký; train.py rẽ nhánh theo config["dataset_type"] == "blend"."""
from detectors import DETECTOR
from .resnet18_detector import ResNet18Detector


@DETECTOR.register_module(module_name="sbi")
class SBIDetector(ResNet18Detector):
    pass
