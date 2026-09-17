"""SBIDataset — học theo DeepfakeBench/training/dataset/sbi_dataset.py:
chỉ giữ ảnh THẬT; mỗi phần tử trả về một CẶP {"fake": (ảnh giả tự trộn, 1), "real": (ảnh thật, 0)};
collate_fn nối thành lô [thật..., giả...]. Mô hình vì thế CHƯA TỪNG thấy deepfake thật khi huấn luyện —
nếu nó vẫn nhận ra deepfake thật ở tập test thì thứ nó học là dấu vết trộn ảnh, không phải đặc điểm của một bộ dữ liệu.
(Lưu ý: bản của họ khai báo self.transform nhưng __getitem__ không dùng; ta cũng chỉ dùng biến đổi bên trong SBI_API.)"""
import torch

from .wdf_dataset import WDFDataset
from .sbi_api import SBI_API


class SBIDataset(WDFDataset):
    def __init__(self, config, mode="train", splits=None, base=None):
        if base is None:
            super().__init__(config, mode, splits)
        else:                                            # dựng từ một WDFDataset đã chia train/val
            self.__dict__.update(base.__dict__)
        keep = [i for i, l in enumerate(self.label_list) if l == 0]
        self.image_list = [self.image_list[i] for i in keep]; self.name_list = [self.name_list[i] for i in keep]
        self.label_list = [0] * len(keep); self.data_dict = {"image": self.name_list, "label": self.label_list}
        self.sbi = SBI_API("train", self.res, config.get("manualSeed", 0))

    def __getitem__(self, i):
        img_f, img_r, _ = self.sbi(self.image_list[i])
        return {"fake": (self.normalize(self.to_tensor(img_f)), 1),
                "real": (self.normalize(self.to_tensor(img_r)), 0), "name": self.name_list[i]}

    @staticmethod
    def collate_fn(batch):
        fi, fl = zip(*[b["fake"] for b in batch]); ri, rl = zip(*[b["real"] for b in batch])
        return {"image": torch.cat([torch.stack(ri), torch.stack(fi)]),
                "label": torch.tensor(list(rl) + list(fl), dtype=torch.long),
                "name": [b["name"] for b in batch] * 2}
