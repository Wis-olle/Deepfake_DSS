"""Bộ dữ liệu WildDeepfake — học theo DeepfakeAbstractBaseDataset (DeepfakeBench/training/dataset/abstract_dataset.py).
Giữ nguyên hình dạng: collect_img_and_label → data_dict{'image','label'}; load_rgb; init_data_aug_method; data_aug;
to_tensor; normalize; __getitem__ trả dict; collate_fn tĩnh trả data_dict.
Bỏ: lmdb, landmark, mask, video_mode, 13 bộ FF++ (ta chỉ có WildDeepfake — khuôn mặt đã cắt sẵn).
Thêm: chia validation THEO VIDEO (họ không có validation: chọn checkpoint ngay trên tập test — ta không làm thế).
Ảnh nạp một lần vào RAM (≈ 3 nghìn khung 96 px) rồi tăng cường bằng PIL từng phần tử — không cần albumentations/OpenCV."""
import copy, io, pathlib
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils import data

EXT = (".png", ".jpg", ".jpeg")


class WDFDataset(data.Dataset):
    def __init__(self, config, mode="train", splits=None):
        self.config, self.mode = config, mode
        self.res, self.frame_num = config["resolution"], config["frame_num"][mode]
        self.root = pathlib.Path(config["rgb_dir"])
        if splits is None:
            splits = ["real_train", "fake_train"] if mode == "train" else ["real_test", "fake_test"]
        self.image_list, self.label_list, self.name_list = [], [], []
        for s in splits:
            self.collect_img_and_label_for_one_split(s)
        assert self.image_list, f"không nạp được ảnh nào cho {mode}: {self.root} / {splits}"
        self.data_dict = {"image": self.name_list, "label": self.label_list}
        self.rng = np.random.default_rng(config.get("manualSeed", 0))
        self.transform = self.init_data_aug_method() if mode == "train" and config.get("use_data_augmentation", True) else None

    # ---------- gom ảnh ----------
    def collect_img_and_label_for_one_split(self, split):
        label = self.config["label_dict"][split.split("_")[0]]          # real_* → 0, fake_* → 1
        for v in sorted((self.root / split).iterdir()):
            frames = sorted(q for q in v.rglob("*") if q.suffix.lower() in EXT)
            if not frames:
                continue
            for i in np.unique(np.linspace(0, len(frames) - 1, self.frame_num).astype(int)):   # frame_num khung rải đều
                self.image_list.append(self.load_rgb(frames[i])); self.label_list.append(label)
                self.name_list.append(f"{split}/{v.name}/{i}")                                   # metrics gom theo 'split/video'

    def load_rgb(self, path):
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB").resize((self.res, self.res), Image.BILINEAR))

    # ---------- tăng cường: cùng danh sách phép biến đổi với họ (albumentations) nhưng bằng PIL ----------
    def init_data_aug_method(self):
        a, rng = self.config["data_aug"], self.rng
        def aug(pil):
            if rng.random() < a["flip_prob"]:       pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < a["rotate_prob"]:     pil = pil.rotate(float(rng.uniform(*a["rotate_limit"])), Image.BILINEAR)
            if rng.random() < a["blur_prob"]:       pil = pil.filter(ImageFilter.GaussianBlur(float(rng.uniform(*a["blur_limit"]))))
            if rng.random() < a["brightness_prob"]:
                pil = ImageEnhance.Brightness(pil).enhance(1 + float(rng.uniform(*a["brightness_limit"])))
                pil = ImageEnhance.Contrast(pil).enhance(1 + float(rng.uniform(*a["contrast_limit"])))
            if rng.random() < 0.5:                  # ImageCompression p=0.5
                buf = io.BytesIO(); pil.save(buf, "JPEG", quality=int(rng.integers(a["quality_lower"], a["quality_upper"] + 1)))
                pil = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
            return pil
        return aug

    def data_aug(self, img):
        return np.asarray(self.transform(Image.fromarray(img))) if self.transform else img

    def to_tensor(self, img):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.     # np.array = bản sao ghi được

    def normalize(self, t):
        m = torch.tensor(self.config["mean"]).view(3, 1, 1); s = torch.tensor(self.config["std"]).view(3, 1, 1)
        return (t - m) / s

    def __getitem__(self, i):
        img = self.data_aug(self.image_list[i]) if self.mode == "train" else self.image_list[i]
        return {"image": self.normalize(self.to_tensor(img)), "label": self.label_list[i], "name": self.name_list[i]}

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        return {"image": torch.stack([b["image"] for b in batch]),
                "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
                "name": [b["name"] for b in batch]}

    # ---------- phần ta thêm: thao tác theo video ----------
    def videos(self):
        return list(dict.fromkeys(n.rsplit("/", 1)[0] for n in self.name_list))

    def subset(self, keep, mode=None):
        """Bản sao nông chỉ giữ các video trong `keep`; mode='test' tắt tăng cường (dùng cho validation)."""
        keep = set(keep); ds = copy.copy(self)
        idx = [i for i, n in enumerate(self.name_list) if n.rsplit("/", 1)[0] in keep]
        ds.image_list = [self.image_list[i] for i in idx]; ds.label_list = [self.label_list[i] for i in idx]
        ds.name_list = [self.name_list[i] for i in idx]; ds.data_dict = {"image": ds.name_list, "label": ds.label_list}
        if mode:
            ds.mode = mode; ds.transform = ds.transform if mode == "train" else None
        return ds


def split_train_val(ds, frac=0.2, seed=0, mode="cluster"):
    """cluster: validation = một cụm video KHÁC train về màu / độ sáng (không dùng chung lối tắt với train).
    random: bốc ngẫu nhiên theo video, giữ tỉ lệ lớp. Trả về (train, val, mô tả)."""
    vids = ds.videos(); feat, lab = {}, {}
    for img, name, l in zip(ds.image_list, ds.name_list, ds.label_list):
        v = name.rsplit("/", 1)[0]; lab[v] = l
        feat.setdefault(v, []).append(np.r_[img.reshape(-1, 3).mean(0) / 255., img.std() / 255.])
    labs = np.array([lab[v] for v in vids]); rng = np.random.default_rng(seed); va = None
    if mode == "cluster":
        from sklearn.cluster import KMeans
        F = np.stack([np.mean(feat[v], 0) for v in vids])
        km = KMeans(n_clusters=5, n_init=10, random_state=seed).fit(F); best = None
        for c in range(5):
            m = km.labels_ == c; n0, n1 = int((labs[m] == 0).sum()), int((labs[m] == 1).sum())
            if min(n0, n1) < 3:                                   # cụm phải có đủ cả hai lớp
                continue
            if best is None or abs(m.mean() - frac) < best[0]:
                best = (abs(m.mean() - frac), c, n0, n1)
        if best:
            va = [v for v, m in zip(vids, km.labels_ == best[1]) if m]
            info = f"cụm màu #{best[1]}: {len(va)} video ({best[2]} thật, {best[3]} giả), khác train về màu / độ sáng"
    if va is None:
        va = []
        for l in (0, 1):
            vs = [v for v, y in zip(vids, labs) if y == l]; rng.shuffle(vs); va += vs[:max(1, int(len(vs) * frac))]
        info = f"ngẫu nhiên theo video: {len(va)} video"
    tr = [v for v in vids if v not in set(va)]
    return ds.subset(tr), ds.subset(va, mode="test"), info
