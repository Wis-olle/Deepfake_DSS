"""
Tải kho ứng viên WildDeepfake (Zi et al., ACM MM 2020) từ HuggingFace, đo độ sáng từng video,
rồi chọn tập con sao cho hai lớp thật / giả có CÙNG phân phối độ sáng — sửa lệch nguồn.

  python get_wdf.py                 # tải pool → wdf_pool/, chọn khớp → wdf/, in báo cáo trước / sau
  python get_wdf.py --no-download   # chỉ chọn lại từ pool đã có

Tập con cũ (lệch nguồn, dùng trong sổ tay) được giữ ở wdf_v1/.
Vì sao phải khớp: nếu video thật tối hơn video giả một cách hệ thống, mạng học "tối = thật"
thay vì học dấu vết giả mạo (xem Sổ tay Deepfake, Bước 5 và Bước 9).
"""
import argparse, collections, io, json, os, pathlib, random, shutil, sys, tarfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
from PIL import Image

REPO = "xingjunm/WildDeepfake"
POOL, OUT, V1 = pathlib.Path("wdf_pool"), pathlib.Path("wdf"), pathlib.Path("wdf_v1")
PLAN = {"real_train": (90, 15), "fake_train": (90, 15), "real_test": (42, 60), "fake_test": (60, 60)}  # (số tệp, trần MB)
TARGET = {"train": 50, "test": 25}     # số video mỗi lớp sau khi khớp
BIN = 32                               # bề rộng dải độ sáng để khớp (thang 0–255)
EXT = (".png", ".jpg", ".jpeg")


def frames_of(v):
    return sorted(q for q in v.rglob("*") if q.suffix.lower() in EXT)


# ---------- 1. tải kho ứng viên ----------
def download():
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(REPO, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings if s.rfilename.endswith(".tar.gz")]
    rng = random.Random(0); total = 0
    for split, (n, cap) in PLAN.items():
        cand = sorted(f for f, sz in files if f.split("/")[1] == split and sz < cap * 2**20)
        rng.shuffle(cand)
        for f in cand[:n]:
            dest = POOL / split / pathlib.Path(f).name.replace(".tar.gz", "")
            if dest.exists():
                continue
            p = hf_hub_download(REPO, f, repo_type="dataset"); total += pathlib.Path(p).stat().st_size
            dest.mkdir(parents=True)
            with tarfile.open(p) as t:
                t.extractall(dest)
        print(f"  {split:11s}: {len(list((POOL / split).iterdir()))} video trong pool (trần {cap} MB)")
    print(f"  tải mới {total / 2**20:.0f} MB")


# ---------- 2. đo từng video ----------
def measure(v, k=8):
    fr = frames_of(v)
    if not fr:
        return None
    idx = np.unique(np.linspace(0, len(fr) - 1, k).astype(int))
    arr = np.stack([np.asarray(Image.open(fr[i]).convert("RGB").resize((64, 64))) for i in idx]).astype(np.float32)
    return dict(bright=float(arr.mean()), rgb=[float(x) for x in arr.mean(axis=(0, 1, 2))], n=len(fr))


def stats_all():
    cache = POOL / "stats.json"
    st = json.loads(cache.read_text()) if cache.exists() else {}
    for split in PLAN:
        for v in sorted((POOL / split).iterdir()):
            key = f"{split}/{v.name}"
            if key not in st:
                m = measure(v)
                if m: st[key] = m
    cache.write_text(json.dumps(st, indent=1))
    return st


# ---------- 3. khớp phân phối độ sáng giữa hai lớp ----------
def match(st, real, fake, target, rng):
    """Ghép cặp thật–giả trong cùng dải độ sáng → hai lớp có histogram độ sáng giống hệt nhau."""
    by = {"r": collections.defaultdict(list), "f": collections.defaultdict(list)}
    for v in real: by["r"][int(st[v]["bright"] // BIN)].append(v)
    for v in fake: by["f"][int(st[v]["bright"] // BIN)].append(v)
    pairs = []
    for b in sorted(set(by["r"]) | set(by["f"])):
        r, f = by["r"][b], by["f"][b]; rng.shuffle(r); rng.shuffle(f)
        pairs += list(zip(r, f))                     # chỉ lấy tới số nhỏ hơn của hai bên
    rng.shuffle(pairs); pairs = pairs[:target]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def link_tree(src, dst):
    for q in frames_of(src):
        d = dst / q.relative_to(src); d.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(q, d)                            # hard link: không tốn thêm đĩa
        except OSError:
            shutil.copy2(q, d)


def report(st, ids, label):
    b = np.array([st[v]["bright"] for v in ids]); rgb = np.array([st[v]["rgb"] for v in ids])
    return f"{label:28s} n={len(ids):3d}  độ sáng {b.mean():6.1f} ± {b.std():4.1f}   RGB {rgb.mean(0).round(0)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-download", action="store_true"); args = ap.parse_args()
    if not args.no_download:
        print("=== 1. Tải kho ứng viên ==="); download()
    print("=== 2. Đo độ sáng từng video ==="); st = stats_all(); print(f"  {len(st)} video đã đo")
    rng = random.Random(0); sel = {}
    print("=== 3. Chọn tập con khớp độ sáng ===")
    for part in ("train", "test"):
        real = [k for k in st if k.startswith(f"real_{part}/")]; fake = [k for k in st if k.startswith(f"fake_{part}/")]
        print("  TRƯỚC  " + report(st, real, f"real_{part} (pool)")); print("  TRƯỚC  " + report(st, fake, f"fake_{part} (pool)"))
        r_sel, f_sel = match(st, real, fake, TARGET[part], rng)
        print("  SAU    " + report(st, r_sel, f"real_{part} (chọn)")); print("  SAU    " + report(st, f_sel, f"fake_{part} (chọn)"))
        sel[f"real_{part}"] = r_sel; sel[f"fake_{part}"] = f_sel
    if OUT.exists() and not V1.exists():
        OUT.rename(V1); print(f"  tập con cũ (lệch nguồn) → {V1}/")
    elif OUT.exists():
        shutil.rmtree(OUT)
    for split, ids in sel.items():
        for key in ids:
            link_tree(POOL / key, OUT / split / key.split("/")[1])
    (OUT / "selection.json").write_text(json.dumps({k: [x.split("/")[1] for x in v] for k, v in sel.items()}, indent=1))
    n_img = sum(len(frames_of(v)) for s in sel for v in (OUT / s).iterdir())
    print(f"=== Xong: {OUT}/ có {sum(len(v) for v in sel.values())} video, {n_img} ảnh; danh sách ở {OUT}/selection.json ===")
