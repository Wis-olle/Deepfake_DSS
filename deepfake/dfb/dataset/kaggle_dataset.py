"""Miền thứ hai để kiểm tra chéo bộ dữ liệu (cách DeepfakeBench đánh giá: train một bộ, test nhiều bộ khác).
Nguồn: mirror Kaggle "Hemg/deepfake-and-real-images" trên HuggingFace (ảnh đơn, không phải video).
Mỗi ảnh là một 'video' một khung → AUC video = AUC khung."""
import io, json
import numpy as np
from PIL import Image

from .wdf_dataset import WDFDataset


class KaggleDataset(WDFDataset):
    def __init__(self, config, n_per_class=800):
        self.config, self.mode, self.res, self.transform = config, "test", config["resolution"], None
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        t = pq.read_table(hf_hub_download("Hemg/deepfake-and-real-images", "data/train-00000-of-00005.parquet", repo_type="dataset"))
        names = None
        try:
            names = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
        except Exception:
            pass
        fake_id = names.index("Fake") if names and "Fake" in names else 0       # nhãn gốc: 0 = Fake, 1 = Real
        self.image_list, self.label_list, self.name_list = [], [], []
        cnt = {0: 0, 1: 0}
        for batch in t.to_batches(max_chunksize=512):
            for row in batch.to_pylist():
                lab = 1 if row["label"] == fake_id else 0
                if cnt[lab] >= n_per_class:
                    continue
                im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").resize((self.res, self.res), Image.BILINEAR)
                self.image_list.append(np.asarray(im)); self.label_list.append(lab)
                self.name_list.append(f"kaggle/{len(self.name_list)}/0"); cnt[lab] += 1
            if cnt[0] >= n_per_class and cnt[1] >= n_per_class:
                break
        self.data_dict = {"image": self.name_list, "label": self.label_list}
