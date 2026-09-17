"""
Nhận biết deepfake bằng mạng tích chập — dựng từ đầu, rồi transfer learning, rồi kiểm tra ngoài miền.
Dữ liệu: WildDeepfake (Zi et al., ACM MM 2020) — khuôn mặt đã cắt sẵn từ deepfake thật trên internet,
         bản mirror HuggingFace xingjunm/WildDeepfake, tải một tập con bằng get_wdf.py.
Quy trình bám "A Recipe for Training Neural Networks" (Karpathy, 2019), giống train_credit_mlp.py.

Cài:   pip install torch torchvision pillow scikit-learn matplotlib pyarrow huggingface_hub
Chạy:  python get_wdf.py                 # tải ~800 MB, tạo thư mục wdf/
       python train_deepfake_cnn.py      # ~15 phút CPU ở 96×96; nhanh hơn nhiều nếu có GPU
Kết quả: out/deepfake_results.md, out/curves_cnn.png, out/worst.png, out/gradcam.png
"""
import argparse, io, json, os, sys, time, pathlib, collections
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT, OUT, IMG, K = pathlib.Path("wdf"), pathlib.Path("out"), 96, 24   # 96 px, tối đa 24 khung / video
OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# Ma trận chi phí của QUYẾT ĐỊNH kiểm duyệt: để lọt một deepfake đắt gấp 3 lần gỡ nhầm một video thật.
# ponytail: 3:1 là giả định minh hoạ — nền tảng thật có số riêng.
C_MISS, C_FALSE = 3.0, 1.0
EXT = (".png", ".jpg", ".jpeg")


# ============ BƯỚC 1–2. NẠP THEO VIDEO, CHIA THEO VIDEO ============
def load_split(split):
    """Trả về (ảnh uint8 [N,3,IMG,IMG], nhãn, id video, độ phân giải gốc)."""
    X, y, vid, res = [], [], [], []
    lab = 1 if split.startswith("fake") else 0
    for v in sorted((ROOT / split).iterdir()):
        frames = sorted(q for q in v.rglob("*") if q.suffix.lower() in EXT)
        if not frames: continue
        for i in np.unique(np.linspace(0, len(frames) - 1, K).astype(int)):
            with Image.open(frames[i]) as im:
                res.append(min(im.size)); im = im.convert("RGB").resize((IMG, IMG), Image.BILINEAR)
                X.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1))
            y.append(lab); vid.append(f"{split}/{v.name}")
    return torch.stack(X), np.array(y), np.array(vid), np.array(res)


J = {}
def look(Xtr, ytr, vtr, rtr):
    print("=== BƯỚC 1: Nhìn dữ liệu ===")
    nv = collections.Counter(zip(vtr, ytr))
    print(f"  khung huấn luyện: {len(ytr)} | video: {len(nv)} | tỉ lệ khung fake: {ytr.mean():.3f}")
    print(f"  độ phân giải gốc (cạnh ngắn) — thật: median {np.median(rtr[ytr==0]):.0f}px, giả: median {np.median(rtr[ytr==1]):.0f}px")
    bright = Xtr.float().mean(dim=(1, 2, 3)).numpy()
    J.update(bright_real=float(bright[ytr==0].mean()), bright_fake=float(bright[ytr==1].mean()), frames=int(len(ytr)))
    print(f"  độ sáng trung bình — thật: {bright[ytr==0].mean():.1f}, giả: {bright[ytr==1].mean():.1f}  (nếu lệch nhiều: mạng có thể học 'độ sáng' thay vì 'khuôn mặt')")
    pass


def save_grid(parts, fname="wdf_grid.png"):
    """Lưới 4 hàng × 8 ảnh: khung giữa của 8 video đầu mỗi phần — NHÌN trước khi mô hình hoá."""
    fig, ax = plt.subplots(4, 8, figsize=(12, 6.4))
    for r, (name, X, vid) in enumerate(parts):
        vids = list(dict.fromkeys(vid))
        for c in range(8):
            idx = np.where(vid == vids[c % len(vids)])[0]; i = idx[len(idx) // 2]
            ax[r, c].imshow(X[i].permute(1, 2, 0).numpy()); ax[r, c].axis("off")
        ax[r, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=100); plt.close(fig)
    print(f"  lưới ảnh mẫu: {OUT / fname} — NHÌN nó trước khi làm gì tiếp")


def split_val(y, vid, frac=0.2, seed=0, X=None, mode="cluster"):
    """mode='cluster': validation = một cụm video KHÁC train về màu / độ sáng (tách theo 'nguồn' gần đúng),
    để validation không dùng chung lối tắt với train. mode='random': tách ngẫu nhiên theo video (cách cũ)."""
    rng = np.random.default_rng(seed)
    vids = np.unique(vid); lab = np.array([y[vid == v][0] for v in vids])
    if mode == "cluster" and X is not None:
        from sklearn.cluster import KMeans
        feat = np.stack([np.r_[X[vid == v].float().mean(dim=(0, 2, 3)).numpy() / 255., X[vid == v].float().std().item() / 255.] for v in vids])
        km = KMeans(n_clusters=5, n_init=10, random_state=seed).fit(feat)
        best = None
        for c in range(5):
            m = km.labels_ == c; n0, n1 = int((lab[m] == 0).sum()), int((lab[m] == 1).sum())
            if min(n0, n1) < 3: continue                                  # cụm phải có đủ cả hai lớp
            score = abs(m.mean() - frac)
            if best is None or score < best[0]: best = (score, c, n0, n1)
        if best is not None:
            va = vids[km.labels_ == best[1]]
            print(f"  validation = cụm màu #{best[1]}: {len(va)} video ({best[2]} thật, {best[3]} giả) — khác train về màu / độ sáng")
            return ~np.isin(vid, va), np.isin(vid, va)
        print("  không có cụm nào đủ cả hai lớp → tách ngẫu nhiên")
    tr, va = [], []
    for l in (0, 1):
        vs = vids[lab == l].copy(); rng.shuffle(vs); nva = max(1, int(len(vs) * frac))
        va += list(vs[:nva]); tr += list(vs[nva:])
    return np.isin(vid, tr), np.isin(vid, va)


# ============ ĐO LƯỜNG ============
def video_auc(y, p, vid):
    df = collections.defaultdict(list)
    for yi, pi, vi in zip(y, p, vid): df[vi].append((yi, pi))
    yv = np.array([v[0][0] for v in df.values()]); pv = np.array([np.mean([x[1] for x in v]) for v in df.values()])
    return roc_auc_score(yv, pv), yv, pv, list(df.keys())


def cost_at(yv, pv, thr):
    pred = pv >= thr
    return (C_MISS * ((pred == 0) & (yv == 1)).sum() + C_FALSE * ((pred == 1) & (yv == 0)).sum()) / len(yv)


# ============ BƯỚC 3. BASELINE ============
def baselines(Xtr, ytr, Xte, yte, vte):
    print("=== BƯỚC 3: Baseline ===")
    print(f"  đa số (luôn đoán 'thật'): AUC 0.500")
    def small_gray(X): return F.interpolate(X.float().mean(1, keepdim=True), size=32).flatten(1).numpy() / 255.
    lr = LogisticRegression(max_iter=2000, C=0.1).fit(small_gray(Xtr), ytr)
    p = lr.predict_proba(small_gray(Xte))[:, 1]
    print(f"  hồi quy logistic trên 32×32 điểm ảnh xám: AUC khung {roc_auc_score(yte, p):.3f} | AUC video {video_auc(yte, p, vte)[0]:.3f}")
    def spectrum(X):   # Durall 2020 / Frank 2020: phổ tần số 1-D (trung bình theo bán kính) của ảnh xám
        g = X.float().mean(1) / 255.; f = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs().log1p()
        c = IMG // 2; yy, xx = torch.meshgrid(torch.arange(IMG), torch.arange(IMG), indexing="ij")
        r = ((yy - c) ** 2 + (xx - c) ** 2).sqrt().long().clamp(max=c - 1)
        out = torch.zeros(len(X), c)
        for k in range(c): out[:, k] = f[:, r == k].mean(1)
        return out.numpy()
    lr2 = LogisticRegression(max_iter=2000).fit(spectrum(Xtr), ytr)
    p2 = lr2.predict_proba(spectrum(Xte))[:, 1]
    print(f"  hồi quy logistic trên phổ tần số 1-D (48 số):   AUC khung {roc_auc_score(yte, p2):.3f} | AUC video {video_auc(yte, p2, vte)[0]:.3f}")
    # Máy dò lối tắt: mô hình chỉ thấy 1 số (độ sáng) hoặc 3 số (RGB trung bình) — không thể thấy dấu vết giả mạo.
    # Nếu nó vẫn đạt AUC cao, hai lớp khác nhau ở thứ không liên quan tới giả mạo (lệch nguồn). 0.5 = đã xoá.
    def bright1(X): return X.float().mean(dim=(1, 2, 3)).numpy()[:, None] / 255.
    def rgb3(X): return X.float().mean(dim=(2, 3)).numpy() / 255.
    for name, key, fn in (("1 số: độ sáng", "bright", bright1), ("3 số: RGB trung bình", "rgb", rgb3)):
        lr3 = LogisticRegression(max_iter=1000).fit(fn(Xtr), ytr); p3 = lr3.predict_proba(fn(Xte))[:, 1]
        print(f"  [dò lối tắt] hồi quy logistic trên {name}: AUC khung {roc_auc_score(yte, p3):.3f} | AUC video {video_auc(yte, p3, vte)[0]:.3f}   (0.5 = lối tắt đã bị xoá)")
        J[f"shortcut_{key}_video"] = float(video_auc(yte, p3, vte)[0])
    J.update(lr_pix_frame=float(roc_auc_score(yte, p)), lr_pix_video=float(video_auc(yte, p, vte)[0]), lr_spec_frame=float(roc_auc_score(yte, p2)), lr_spec_video=float(video_auc(yte, p2, vte)[0]))
    return {"LR điểm ảnh": video_auc(yte, p, vte)[0], "LR phổ tần số": video_auc(yte, p2, vte)[0]}


# ============ BƯỚC 4. MẠNG TÍCH CHẬP NHỎ, TỰ DỰNG ============
def block(cin, cout):   # một "tầng" của CNN: tích chập 3×3 → chuẩn hoá lô → ReLU → gộp cực đại 2×2
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))  # 96→48→24→12→6
        self.head = nn.Linear(128, 1)                                                              # sau gộp trung bình toàn cục
        nn.init.zeros_(self.head.weight)   # lớp cuối = 0 → logit khởi tạo = bias → loss đúng ln2 (không zero-init: 0.94, kiểm tra 1 bắt được)
    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3))).squeeze(-1)

def resnet18():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, 1)
    return m

def norm(xb):  # uint8 → float chuẩn hoá theo thống kê ImageNet
    return ((xb.float() / 255.) - MEAN) / STD

def logit_of(m, x):
    out = m(x); return out.squeeze(-1) if out.dim() > 1 else out


# ============ BƯỚC 5. TĂNG CƯỜNG DỮ LIỆU + SUY GIẢM CHẤT LƯỢNG (bài học NTIRE 2026) ============
def augment(xb, rng, degrade=True, color=False):
    xb = xb.clone()
    flip = torch.from_numpy(rng.random(len(xb)) < 0.5)
    xb[flip] = xb[flip].flip(-1)                                   # lật ngang
    if not degrade and not color: return xb
    out = []
    for im in xb:
        pil = Image.fromarray(im.permute(1, 2, 0).numpy())
        if color and rng.random() < 0.8:   # phá lối tắt độ sáng / màu: xáo độ sáng, tương phản, bão hoà → chúng hết mang thông tin
            pil = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(0.6, 1.4)))
            pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(0.7, 1.3)))
            pil = ImageEnhance.Color(pil).enhance(float(rng.uniform(0.7, 1.3)))
        r = rng.random() if degrade else 1.0
        if r < 0.33:   # nén JPEG chất lượng ngẫu nhiên
            buf = io.BytesIO(); pil.save(buf, "JPEG", quality=int(rng.integers(30, 90))); pil = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        elif r < 0.55: # làm mờ
            pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.5, 1.5))))
        elif r < 0.75: # thu nhỏ rồi phóng lại
            s = int(rng.integers(40, 80)); pil = pil.resize((s, s), Image.BILINEAR).resize((IMG, IMG), Image.BILINEAR)
        out.append(torch.from_numpy(np.asarray(pil)).permute(2, 0, 1))
    return torch.stack(out)


def sanity(Xtr, ytr, seed):
    print("=== BƯỚC 4: Kiểm tra pipeline ===")
    torch.manual_seed(seed); m = SmallCNN().to(DEV)
    x = norm(Xtr[:256]).to(DEV); y = torch.tensor(ytr[:256], dtype=torch.float32, device=DEV)
    with torch.no_grad(): l0 = F.binary_cross_entropy_with_logits(m(x), y).item()
    prior = float(ytr.mean())
    with torch.no_grad(): m.head.bias.fill_(np.log(prior / (1 - prior))); l1 = F.binary_cross_entropy_with_logits(m(x), y).item()
    J.update(init_loss=l0, init_loss_prior=l1)
    print(f"  [kiểm tra 1] loss khởi tạo = {l0:.3f} (≈ ln2 = 0.693) → đặt bias theo prior = {l1:.3f}")
    xs, ys = x[:32], y[:32]; opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(m(xs), ys); loss.backward(); opt.step()
    J.update(overfit_loss=loss.item(), p_small=sum(q.numel() for q in SmallCNN().parameters()), p_resnet=sum(q.numel() for q in torchvision.models.resnet18().parameters()))
    print(f"  [kiểm tra 2] overfit 32 ảnh: loss cuối = {loss.item():.4f} (phải gần 0)")
    print(f"  tham số SmallCNN: {sum(p.numel() for p in SmallCNN().parameters()):,} | ResNet-18: {sum(p.numel() for p in torchvision.models.resnet18().parameters()):,}")


def train(model, Xtr, ytr, Xva, yva, vva, seed, lr=3e-4, wd=1e-4, epochs=25, patience=5, batch=64, aug=True, degrade=True, color=False, tag=""):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = model.to(DEV)
    if hasattr(m, "head"):
        with torch.no_grad(): m.head.bias.fill_(np.log(ytr.mean() / (1 - ytr.mean())))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    yt = torch.tensor(ytr, dtype=torch.float32)
    hist, best, best_state, bad, t0 = [], -1, None, 0, time.time()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr)); tl = 0.
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]; xb = Xtr[idx]
            if aug: xb = augment(xb, rng, degrade, color)
            opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(logit_of(m, norm(xb).to(DEV)), yt[idx].to(DEV)); loss.backward(); opt.step()
            tl += loss.item() * len(idx)
        pv = predict(m, Xva); va, _, _, _ = video_auc(yva, pv, vva)
        hist.append((tl / len(Xtr), va))
        if va > best: best, best_state, bad = va, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(best_state); m.eval()
    J.setdefault("train", {})[tag] = dict(epochs=len(hist), best_val=float(best), seconds=time.time() - t0)
    print(f"  {tag}: dừng epoch {len(hist)}, val AUC video tốt nhất {best:.3f}, {time.time()-t0:.0f}s")
    return m, hist


@torch.no_grad()
def predict(m, X, bs=256):
    m.eval(); out = []
    for i in range(0, len(X), bs): out.append(torch.sigmoid(logit_of(m, norm(X[i:i + bs]).to(DEV))).cpu())
    return torch.cat(out).numpy()


# ============ MIỀN THỨ HAI: ảnh deepfake từ nguồn khác (bản mirror Kaggle trên HF) ============
def load_xdomain(n_per_class=800):
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        p = hf_hub_download("Hemg/deepfake-and-real-images", "data/train-00000-of-00005.parquet", repo_type="dataset")
        t = pq.read_table(p)
        names = None
        try:
            meta = json.loads(t.schema.metadata[b"huggingface"]); names = meta["info"]["features"]["label"]["names"]
        except Exception: pass
        fake_id = names.index("Fake") if names and "Fake" in names else 0
        X, y, cnt = [], [], collections.Counter()
        for batch in t.to_batches(max_chunksize=512):
            for row in batch.to_pylist():
                lab = 1 if row["label"] == fake_id else 0
                if cnt[lab] >= n_per_class: continue
                im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
                X.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1)); y.append(lab); cnt[lab] += 1
            if cnt[0] >= n_per_class and cnt[1] >= n_per_class: break
        print(f"  miền thứ hai: {len(y)} ảnh, nhãn lớp: {names}, fake_id={fake_id}")
        return torch.stack(X), np.array(y)
    except Exception as e:
        print("  không nạp được miền thứ hai:", e); return None, None


# ============ BƯỚC 8–9. PHẦN ĐUÔI + GRAD-CAM ============
def tail(m, Xte, yte, vte, rte, p, tag):
    print(f"=== BƯỚC 8: Phần đuôi ({tag}) ===")
    eps = 1e-6; loss = -(yte * np.log(p + eps) + (1 - yte) * np.log(1 - p + eps)); worst = np.argsort(-loss)[:10]
    fig, ax = plt.subplots(1, 10, figsize=(15, 2.2))
    for a, i in zip(ax, worst):
        a.imshow(Xte[i].permute(1, 2, 0).numpy()); a.set_title(f"y={yte[i]} p={p[i]:.2f}", fontsize=8); a.axis("off")
    fig.suptitle("10 khung hình mô hình sai nặng nhất (tập test)"); fig.tight_layout(); fig.savefig(OUT / "worst.png", dpi=110); plt.close(fig)
    bright = Xte.float().mean(dim=(1, 2, 3)).numpy()
    for name, val in (("độ sáng", bright), ("độ phân giải gốc", rte)):
        q = np.quantile(val, [1/3, 2/3]); parts = np.digitize(val, q)
        rows = [f"{['thấp','vừa','cao'][k]}: n={int((parts==k).sum())} AUC khung={roc_auc_score(yte[parts==k], p[parts==k]):.3f}" for k in range(3) if len(np.unique(yte[parts==k])) == 2]
        J.setdefault("slices", {})[name] = rows
        print(f"  theo {name}: " + " | ".join(rows))
    print(f"  ảnh: {OUT / 'worst.png'}")


def gradcam(m, X, y, p, layer, fname):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda mod, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__("g", go[0]))
    fig, ax = plt.subplots(2, 8, figsize=(14, 3.8))
    for c, i in enumerate(list(np.where(y == 0)[0][:4]) + list(np.where(y == 1)[0][:4])):
        m.zero_grad(); x = norm(X[i:i + 1]).to(DEV).requires_grad_(True)
        z = logit_of(m, x); z.sum().backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True); cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=IMG, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy(); cam /= cam.max() + 1e-8
        img = X[i].permute(1, 2, 0).numpy()
        ax[0, c].imshow(img); ax[0, c].set_title(f"{'thật' if y[i]==0 else 'giả'} p={p[i]:.2f}", fontsize=8); ax[0, c].axis("off")
        ax[1, c].imshow(img); ax[1, c].imshow(cam, cmap="jet", alpha=0.45); ax[1, c].axis("off")
    h1.remove(); h2.remove()
    fig.suptitle("Grad-CAM: mạng nhìn vào đâu khi quyết định (hàng dưới)"); fig.tight_layout(); fig.savefig(OUT / fname, dpi=110); plt.close(fig)
    print(f"  Grad-CAM: {OUT / fname}")


# ============ CHẠY ============
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-resnet", action="store_true"); ap.add_argument("--root", default="wdf"); ap.add_argument("--out", default="out")
    ap.add_argument("--val-split", default="cluster", choices=["cluster", "random"]); args = ap.parse_args()
    ROOT, OUT = pathlib.Path(args.root), pathlib.Path(args.out); OUT.mkdir(exist_ok=True); print(f"dữ liệu: {ROOT}/  →  kết quả: {OUT}/")
    t0 = time.time()
    Xa, ya, va_, ra = [], [], [], []
    for s in ("real_train", "fake_train"):
        X, y, v, r = load_split(s); Xa.append(X); ya.append(y); va_.append(v); ra.append(r)
    Xall, yall, vall, rall = torch.cat(Xa), np.concatenate(ya), np.concatenate(va_), np.concatenate(ra)
    Xb, yb, vb, rb = [], [], [], []
    for s in ("real_test", "fake_test"):
        X, y, v, r = load_split(s); Xb.append(X); yb.append(y); vb.append(v); rb.append(r)
    Xte, yte, vte, rte = torch.cat(Xb), np.concatenate(yb), np.concatenate(vb), np.concatenate(rb)
    save_grid([("real_train", Xa[0], va_[0]), ("fake_train", Xa[1], va_[1]), ("real_test", Xb[0], vb[0]), ("fake_test", Xb[1], vb[1])])
    mtr, mva = split_val(yall, vall, seed=args.seed, X=Xall, mode=args.val_split)
    Xtr, ytr, vtr, rtr = Xall[mtr], yall[mtr], vall[mtr], rall[mtr]
    Xva, yva, vva = Xall[mva], yall[mva], vall[mva]
    look(Xtr, ytr, vtr, rtr)
    print(f"=== BƯỚC 2: Chia theo VIDEO === train {len(np.unique(vtr))} video/{len(ytr)} khung | val {len(np.unique(vva))}/{len(yva)} | test {len(np.unique(vte))}/{len(yte)} (chia sẵn của bộ dữ liệu)")
    res = baselines(Xtr, ytr, Xte, yte, vte)
    sanity(Xtr, ytr, args.seed)
    print("=== BƯỚC 5: Huấn luyện ===")
    runs = [("SmallCNN, không tăng cường", lambda: SmallCNN(), dict(aug=False, lr=1e-3)),
            ("SmallCNN + lật/JPEG/mờ/thu-phóng", lambda: SmallCNN(), dict(aug=True, lr=1e-3)),
            ("SmallCNN + JPEG/mờ/thu-phóng + xáo màu/độ sáng", lambda: SmallCNN(), dict(aug=True, color=True, lr=1e-3))]
    if not args.no_resnet: runs.append(("ResNet-18 tiền huấn luyện ImageNet + tăng cường đủ", resnet18, dict(aug=True, color=True, lr=1e-4, epochs=10, patience=3)))
    Xxd, yxd = load_xdomain()
    models, table, hists = {}, [], {}
    for tag, mk, kw in runs:
        kw = dict(epochs=args.epochs, **kw) if "epochs" not in kw else kw
        m, hist = train(mk(), Xtr, ytr, Xva, yva, vva, args.seed, tag=tag, **kw); models[tag] = m; hists[tag] = hist
        p = predict(m, Xte); fa = roc_auc_score(yte, p); vauc, yv, pv, _ = video_auc(yte, p, vte)
        xd = roc_auc_score(yxd, predict(m, Xxd)) if Xxd is not None else float("nan")
        table.append((tag, fa, vauc, xd)); res[tag] = vauc
        print(f"     → test: AUC khung {fa:.3f} | AUC video {vauc:.3f} | miền thứ hai (nguồn khác) AUC {xd:.3f}")
    # đường học
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for tag, h in hists.items():
        h = np.array(h); ax[0].plot(h[:, 0], label=tag[:28]); ax[1].plot(h[:, 1], label=tag[:28])
    ax[0].set_title("train loss"); ax[1].set_title("val AUC (video)"); ax[1].legend(fontsize=7); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(OUT / "curves_cnn.png", dpi=120); plt.close(fig)
    # BƯỚC 7: ngưỡng quyết định ở mức VIDEO cho mô hình tốt nhất
    best_tag = max(table, key=lambda r: r[2])[0]; m = models[best_tag]
    pva = predict(m, Xva); _, yv_va, pv_va, _ = video_auc(yva, pva, vva)
    pte = predict(m, Xte); _, yv_te, pv_te, _ = video_auc(yte, pte, vte)
    grid = np.linspace(0.05, 0.95, 91); thr = float(grid[np.argmin([cost_at(yv_va, pv_va, t) for t in grid])]); elkan = C_FALSE / (C_FALSE + C_MISS)
    print(f"=== BƯỚC 7: Ngưỡng quyết định ở mức video ({best_tag}; chi phí lọt:gỡ nhầm = {C_MISS:.0f}:{C_FALSE:.0f}) ===")
    for name, t_ in (("0.50", .5), (f"tốt nhất trên val={thr:.2f}", thr), (f"Elkan={elkan:.2f}", elkan)):
        pred = pv_te >= t_
        J.setdefault("thr", []).append(dict(name=name, thr=float(t_), tp=int((pred & (yv_te==1)).sum()), P=int((yv_te==1).sum()), fp=int((pred & (yv_te==0)).sum()), N=int((yv_te==0).sum()), cost=float(cost_at(yv_te, pv_te, t_))))
        print(f"  ngưỡng {name:22s}: bắt {int((pred & (yv_te==1)).sum())}/{int((yv_te==1).sum())} video giả, gỡ nhầm {int((pred & (yv_te==0)).sum())}/{int((yv_te==0).sum())} video thật, chi phí {cost_at(yv_te, pv_te, t_):.3f}/video")
    tail(m, Xte, yte, vte, rte, pte, best_tag)
    layer = m.features[-1][0] if isinstance(m, SmallCNN) else m.layer4
    gradcam(m, Xte, yte, pte, layer, "gradcam.png")
    md = ["| Mô hình | AUC khung (test) | AUC video (test) | AUC miền thứ hai |", "|---|---|---|---|"]
    md += [f"| Đa số | 0.500 | 0.500 | 0.500 |"] + [f"| {k} | — | {v:.3f} | — |" for k, v in res.items() if k.startswith("LR")]
    md += [f"| {t} | {fa:.3f} | {va:.3f} | {xd:.3f} |" for t, fa, va, xd in table]
    (OUT / "deepfake_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    J["runs"] = {tg: dict(frame=float(fa), video=float(va), xd=float(xd)) for tg, fa, va, xd in table}; J["best"] = best_tag; J["seconds"] = time.time() - t0
    (OUT / "deepfake_results.json").write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + "\n".join(md)); print(f"\nXong trong {time.time()-t0:.0f}s trên {DEV}. Xem out/")
