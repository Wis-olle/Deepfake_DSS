"""Self-Blended Images (Shiohara & Yamasaki, CVPR 2022) — học theo DeepfakeBench/training/dataset/sbi_api.py
(mã gốc của Kaede Shiohara, ĐH Tokyo). Cùng bốn bước, cùng tham số:
  1. mặt nạ vùng mặt              (họ: 4 kiểu hull từ 81 landmark dlib — ta: elip / đa giác lồi ngẫu nhiên quanh tâm ảnh,
                                    vì WildDeepfake chỉ cho ảnh mặt đã cắt, không có landmark)
  2. biến đổi màu ảnh nguồn        (RGBShift ±20 p=.3, HSV, sáng/tương phản ±0.1, rồi thu-phóng ×2/×4 HOẶC làm sắc)
  3. dịch / co giãn nguồn + mặt nạ (±3 % ngang, ±1.5 % dọc, 0.95–1/0.95; bỏ ElasticTransform)
  4. trộn động                     (mặt nạ làm mềm hai lần mờ, nhân tỉ lệ ngẫu nhiên trong [.25,.5,.75,1,1,1])
rồi biến đổi CÙNG tham số cho cả cặp (RGBShift, sáng/tương phản ±0.3, JPEG 40–100) như additional_targets của họ.
Viết bằng PIL + numpy thay cho albumentations + OpenCV. Kết quả: ảnh giả có ĐÚNG một thứ khác ảnh thật — đường
biên trộn và lệch thống kê màu giữa trong/ngoài mặt nạ — nên mạng buộc phải học chính dấu vết đó.
ponytail: mặt nạ hình học thay landmark — có dlib 81 điểm thì thay face_mask() bằng random_get_hull() của họ."""
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

BLEND_LIST = [0.25, 0.5, 0.75, 1, 1, 1]


def face_mask(h, w, rng):
    cx, cy = w * (0.5 + rng.uniform(-0.04, 0.04)), h * (0.5 + rng.uniform(-0.03, 0.07))
    rx, ry = w * rng.uniform(0.30, 0.42), h * rng.uniform(0.38, 0.50)
    m = Image.new("L", (w, h), 0); d = ImageDraw.Draw(m)
    if rng.random() < 0.5:                                           # elip trơn
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    else:                                                            # đa giác lồi nhấp nhô — thay cho 4 kiểu hull
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False); jit = rng.uniform(0.85, 1.05, 16)
        d.polygon([(cx + rx * jit[i] * np.cos(a), cy + ry * jit[i] * np.sin(a)) for i, a in enumerate(ang)], fill=255)
    return np.asarray(m, dtype=np.float32) / 255.


def get_blend_mask(mask, rng):
    """Như mã gốc: co nhỏ 75–100 % → mờ nhẹ → chỉ giữ lõi (=max) → mờ mạnh bán kính ngẫu nhiên → chuẩn hoá [0,1].
    Bán kính mờ của họ chỉnh ở 256 px; nhân với h/256 để giữ cùng tỉ lệ ở cỡ ảnh khác."""
    h, w = mask.shape; s = h / 256.
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize((int(w * rng.uniform(0.75, 1.0)), int(h * rng.uniform(0.75, 1.0))))
    m = m.filter(ImageFilter.GaussianBlur(max(0.3, rng.uniform(1, 4) * s)))
    a = np.asarray(m, dtype=np.float32); a = np.where(a >= a.max(), 1.0, 0.0)           # mask_blured[mask_blured<1]=0
    m = Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(0.5, rng.uniform(5, 46) * s)))
    a = np.asarray(m.resize((w, h)), dtype=np.float32); a /= a.max() + 1e-8
    return a[..., None]


def dynamic_blend(source, target, mask, rng):
    m = get_blend_mask(mask, rng) * BLEND_LIST[rng.integers(len(BLEND_LIST))]
    return m * source + (1 - m) * target, m


def randaffine(img, mask, rng):
    """Dịch ±3 % ngang, ±1.5 % dọc, co giãn 0.95–1/0.95 quanh tâm — áp cùng phép cho ảnh nguồn và mặt nạ."""
    h, w = mask.shape
    sc = rng.uniform(0.95, 1 / 0.95); tx, ty = rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.015, 0.015) * h
    # PIL AFFINE: (a,b,c,d,e,f) ánh xạ điểm ĐÍCH (x,y) → điểm nguồn (a·x+b·y+c, d·x+e·y+f)
    coef = (1 / sc, 0, w / 2 - (w / 2 + tx) / sc, 0, 1 / sc, h / 2 - (h / 2 + ty) / sc)
    warp = lambda arr, rs: np.asarray(Image.fromarray(arr).transform((w, h), Image.AFFINE, coef, rs))
    return warp(img, Image.BILINEAR), warp((mask * 255).astype(np.uint8), Image.NEAREST).astype(np.float32) / 255.


def source_transforms(img, rng):
    if rng.random() < 0.3:                                                         # RGBShift
        img = np.clip(img.astype(np.int16) + rng.integers(-20, 21, 3), 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    pil = ImageEnhance.Color(pil).enhance(1 + float(rng.uniform(-0.1, 0.1)))        # HueSaturationValue (nhẹ)
    pil = ImageEnhance.Brightness(pil).enhance(1 + float(rng.uniform(-0.1, 0.1)))   # RandomBrightnessContrast ±0.1
    pil = ImageEnhance.Contrast(pil).enhance(1 + float(rng.uniform(-0.1, 0.1)))
    w, h = pil.size
    if rng.random() < 0.5:                                                         # OneOf: RandomDownScale ×2 / ×4
        r = int(rng.choice([2, 4])); pil = pil.resize((w // r, h // r), Image.NEAREST).resize((w, h), Image.BILINEAR)
    else:                                                                          #        ... hoặc Sharpen
        pil = ImageEnhance.Sharpness(pil).enhance(float(rng.uniform(1.5, 2.5)))
    return np.asarray(pil)


def pair_transforms(img_f, img_r, rng):
    """Cùng tham số cho cả cặp (additional_targets={'image1':'image'} trong mã gốc)."""
    ops = []
    if rng.random() < 0.3: ops.append(("shift", rng.integers(-20, 21, 3)))
    if rng.random() < 0.3: ops.append(("bc", (1 + float(rng.uniform(-0.3, 0.3)), 1 + float(rng.uniform(-0.3, 0.3)))))
    if rng.random() < 0.5: ops.append(("jpeg", int(rng.integers(40, 101))))
    def apply(img):
        for k, v in ops:
            if k == "shift":
                img = np.clip(img.astype(np.int16) + v, 0, 255).astype(np.uint8)
            elif k == "bc":
                img = np.asarray(ImageEnhance.Contrast(ImageEnhance.Brightness(Image.fromarray(img)).enhance(v[0])).enhance(v[1]))
            else:
                buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=v)
                img = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        return img
    return apply(img_f), apply(img_r)


class SBI_API:
    def __init__(self, phase="train", image_size=96, seed=0):
        assert phase == "train", "SBI chỉ dùng khi huấn luyện"
        self.image_size, self.rng = image_size, np.random.default_rng(seed)

    def self_blending(self, img):
        rng = self.rng; h, w = img.shape[:2]
        mask = face_mask(h, w, rng); source = img.copy()
        if rng.random() < 0.5:                       # nửa số lần biến đổi nguồn, nửa số lần biến đổi đích (như mã gốc)
            source = source_transforms(source, rng)
        else:
            img = source_transforms(img, rng)
        source, mask = randaffine(source, mask, rng)
        blended, mask = dynamic_blend(source.astype(np.float32), img.astype(np.float32), mask, rng)
        return img.astype(np.uint8), np.clip(blended, 0, 255).astype(np.uint8), mask[..., 0]

    def __call__(self, img):
        """img: uint8 [H,W,3] ảnh THẬT → (ảnh giả tự trộn, ảnh thật đã biến đổi cặp, mặt nạ trộn)."""
        if self.rng.random() < 0.5:                  # hflip cả cặp
            img = img[:, ::-1]
        img_r, img_f, mask = self.self_blending(np.ascontiguousarray(img))
        img_f, img_r = pair_transforms(img_f, img_r, self.rng)
        return img_f, img_r, mask
