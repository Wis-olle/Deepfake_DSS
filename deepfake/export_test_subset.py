"""Xuất tập test gọn để đưa lên git: với mỗi video trong wdf/real_test và wdf/fake_test, chép đúng K = 24 khung
mà train_deepfake_cnn.py / dfb sẽ dùng (cùng công thức lấy mẫu: np.linspace(0, n-1, K).astype(int)), giữ nguyên
đường dẫn tương đối, vào wdf_test/. Kết quả: 42 video, 1.008 khung, ~57 MB thay vì 1,3 GB.

    python export_test_subset.py            # wdf/ -> wdf_test/
    python export_test_subset.py --root wdf_v1 --out wdf_v1_test

Dùng lại: python train_deepfake_cnn.py --root wdf_test chỉ để đánh giá (không có tập học trong đó).
"""
import argparse, pathlib, shutil
import numpy as np

EXT = (".png", ".jpg", ".jpeg")      # như train_deepfake_cnn.py
K = 24

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="wdf"); ap.add_argument("--out", default="wdf_test")
    a = ap.parse_args()
    root, out = pathlib.Path(a.root), pathlib.Path(a.out)
    n_video = n_frame = size = 0
    for split in ("real_test", "fake_test"):
        for v in sorted((root / split).iterdir()):
            frames = sorted(q for q in v.rglob("*") if q.suffix.lower() in EXT)
            if not frames: continue
            for i in np.unique(np.linspace(0, len(frames) - 1, K).astype(int)):
                src = frames[i]; dst = out / src.relative_to(root)
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
                n_frame += 1; size += src.stat().st_size
            n_video += 1
    (out / "README.md").write_text(
        "# wdf_test: tập test gọn của WildDeepfake\n\n"
        f"{n_video} video (21 thật + 21 giả), {n_frame} khung = đúng 24 khung/video mà mã lấy mẫu, chép từ `wdf/real_test` và "
        "`wdf/fake_test` bằng `export_test_subset.py`. Nguồn: WildDeepfake (Zi et al., ACM MM 2020), bản mirror "
        "huggingface.co/datasets/xingjunm/WildDeepfake, chỉ dùng cho nghiên cứu; không phân phối lại ngoài phạm vi nhóm.\n\n"
        "Tập học (real_train/fake_train) và kho ứng viên không có ở đây: chạy `python get_wdf.py` để tải lại.\n", encoding="utf-8")
    print(f"{n_video} video, {n_frame} khung, {size/1e6:.1f} MB -> {out}/")

if __name__ == "__main__":
    main()
