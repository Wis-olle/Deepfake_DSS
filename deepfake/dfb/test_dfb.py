"""Kiểm tra tối thiểu — chạy:  ..\\..\\venv\\Scripts\\python.exe test_dfb.py   (không cần dữ liệu, < 30 s)
1. loss khởi tạo SmallCNN = ln2      2. overfit được 8 ảnh      3. SBI sinh cặp khác nhau, mặt nạ hợp lệ
4. AUC video gom đúng theo 'split/video' (tên video trùng giữa real_test/ và fake_test/ không bị trộn)"""
import math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, torch, yaml

from detectors import DETECTOR
from dataset.sbi_api import SBI_API
from metrics.utils import get_test_metrics

HERE = pathlib.Path(__file__).resolve().parent
cfg = yaml.safe_load(open(HERE / "config" / "detector" / "small_cnn.yaml", encoding="utf-8"))
torch.manual_seed(0)


def test_init_loss_is_ln2():
    m = DETECTOR["small_cnn"](cfg); x = torch.randn(16, 3, 96, 96); y = torch.randint(0, 2, (16,))
    loss = m.get_losses({"label": y}, m({"image": x}))["overall"].item()
    assert abs(loss - math.log(2)) < 1e-5, loss


def test_overfit_tiny_batch():
    m = DETECTOR["small_cnn"](cfg); x = torch.randn(8, 3, 96, 96); y = torch.tensor([0, 1] * 4)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(60):
        opt.zero_grad(); loss = m.get_losses({"label": y}, m({"image": x}))["overall"]; loss.backward(); opt.step()
    assert loss.item() < 0.05, loss.item()


def test_sbi_pair():
    rng = np.random.default_rng(0)
    img = np.kron(rng.integers(0, 256, (12, 12, 3)), np.ones((8, 8, 1))).astype(np.uint8)   # ảnh ô vuông có kết cấu
    api = SBI_API("train", 96, seed=0)
    for _ in range(5):
        f, r, mask = api(img)
        assert f.shape == r.shape == (96, 96, 3) and f.dtype == np.uint8 and mask.shape == (96, 96)
        assert 0 <= mask.min() and mask.max() <= 1 + 1e-6 and mask.max() >= 0.25 - 1e-6, "mặt nạ phải trong [0,1] và có lõi = tỉ lệ trộn"
        d = np.abs(f.astype(int) - r.astype(int)).sum(-1)
        inside, outside = d[mask > 0.5 * mask.max()].mean(), d[mask <= np.quantile(mask, 0.1)].mean()   # lõi vs 10 % rìa mờ nhất
        assert inside > 3 and inside > 3 * outside, f"giả phải khác thật TRONG mặt nạ ({inside:.1f}) và giống NGOÀI mặt nạ ({outside:.1f})"


def test_video_auc_groups_by_split_and_video():
    names = ["real_test/101/0", "real_test/101/1", "fake_test/101/0", "fake_test/101/1"]   # cùng tên video 101
    m = get_test_metrics(np.array([0.2, 0.4, 0.9, 0.1]), np.array([0, 0, 1, 1]), names)
    assert len(m["video_names"]) == 2 and m["video_auc"] == 1.0, m       # video thật 0.3 < video giả 0.5
    assert abs(m["auc"] - 0.5) < 1e-9                                     # AUC khung: 2/4 cặp đúng


if __name__ == "__main__":
    for fn in (test_init_loss_is_ln2, test_overfit_tiny_batch, test_sbi_pair, test_video_auc_groups_by_split_and_video):
        fn(); print("OK", fn.__name__)
