"""
Thí nghiệm phụ: ĐƯỜNG HỌC THEO LƯỢNG DỮ LIỆU — "AUC theo số video huấn luyện".
Câu hỏi: thêm dữ liệu có giúp không, và đường cong phẳng ra ở đâu?

Cách làm: dùng lại nguyên pipeline của train_deepfake_cnn.py (nạp theo video, chia validation,
mô hình, vòng huấn luyện), chỉ đổi ĐÚNG MỘT THỨ — số VIDEO huấn luyện mỗi lớp.
Validation được chia MỘT LẦN và giữ nguyên cho mọi kích thước (so sánh mới công bằng);
tập con nhỏ nằm gọn trong tập con lớn của cùng seed; đo AUC mức VIDEO trên tập test chính thức.

Chạy:  ..\venv\Scripts\python.exe data_scaling.py --root wdf_v1 --out out_scaling
       ..\venv\Scripts\python.exe data_scaling.py --sizes 8,16,24,32 --seeds 3 --model resnet --color
Kết quả: <out>/scaling.json (mọi con số), <out>/scaling.png (đường học)
"""
import argparse, json, pathlib, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import train_deepfake_cnn as T


def load(splits):
    """Nạp và nối nhiều split thành (ảnh, nhãn, id video)."""
    X, y, v = [], [], []
    for s in splits:
        a, b, c, _ = T.load_split(s); X.append(a); y.append(b); v.append(c)
    return torch.cat(X), np.concatenate(y), np.concatenate(v)


ap = argparse.ArgumentParser(description="Đường học: AUC mức video theo số video huấn luyện mỗi lớp")
ap.add_argument("--root", default="wdf", help="thư mục dữ liệu (chứa real_train/, fake_train/, ...)")
ap.add_argument("--out", default="out", help="thư mục kết quả")
ap.add_argument("--sizes", default="8,16,24,32", help="số video MỖI LỚP, cách nhau bởi dấu phẩy")
ap.add_argument("--seeds", type=int, default=2, help="số lần lặp lại (mỗi lần một cách bốc video khác)")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--model", default="small", choices=["small", "resnet"])
ap.add_argument("--color", action="store_true", help="bật tăng cường xáo màu / độ sáng")
args = ap.parse_args()

T.ROOT = pathlib.Path(args.root)                       # load_split đọc theo biến toàn cục này
OUT = pathlib.Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
sizes = sorted({int(s) for s in args.sizes.split(",") if s.strip()})
t0 = time.time()
print(f"dữ liệu: {T.ROOT}/  →  kết quả: {OUT}/  | mô hình: {args.model} | kích thước: {sizes} | số seed: {args.seeds} | thiết bị: {T.DEV}")

Xall, yall, vall = load(("real_train", "fake_train"))
Xte, yte, vte = load(("real_test", "fake_test"))
mtr, mva = T.split_val(yall, vall, frac=0.2, seed=0, mode="random")     # chia MỘT LẦN, giữ cố định
Xva, yva, vva = Xall[mva], yall[mva], vall[mva]
Xpool, ypool, vpool = Xall[mtr], yall[mtr], vall[mtr]                   # kho video để bốc ra huấn luyện
vids = np.unique(vpool); lab = np.array([ypool[vpool == q][0] for q in vids])
pool = {0: vids[lab == 0], 1: vids[lab == 1]}
va_set = set(np.unique(vva))
print(f"kho train: {len(pool[0])} video thật + {len(pool[1])} video giả"
      f" | val {len(va_set)} video/{len(yva)} khung | test {len(np.unique(vte))} video/{len(yte)} khung")

cap = min(len(pool[0]), len(pool[1]))
qua_lon = [n for n in sizes if n > cap]
if qua_lon: print(f"  (bỏ qua kích thước {qua_lon}: kho chỉ có {cap} video mỗi lớp)")
sizes = [n for n in sizes if n <= cap] or [cap]

kw = dict(lr=1e-3, epochs=args.epochs) if args.model == "small" else dict(lr=1e-4, epochs=10, patience=3)
runs = []
for seed in range(args.seeds):
    rng = np.random.default_rng(seed)
    order = {l: rng.permutation(pool[l]) for l in (0, 1)}   # xáo một lần → tập con nhỏ nằm trong tập con lớn
    for n in sizes:
        pick = np.concatenate([order[0][:n], order[1][:n]])
        assert len(set(pick)) == 2 * n and not (set(pick) & va_set), "bốc video sai: trùng lặp hoặc rò rỉ validation"
        sub = np.isin(vpool, pick); Xs, ys = Xpool[sub], ypool[sub]
        tag = f"n={n} video/lớp, seed {seed}"
        torch.manual_seed(seed)                             # khởi tạo trọng số lặp lại được
        model = T.SmallCNN() if args.model == "small" else T.resnet18()
        m, _ = T.train(model, Xs, ys, Xva, yva, vva, seed, color=args.color, tag=tag, **kw)
        auc_te = float(T.video_auc(yte, T.predict(m, Xte), vte)[0])
        auc_va = float(T.video_auc(yva, T.predict(m, Xva), vva)[0])
        info = T.J.get("train", {}).get(tag, {})
        runs.append(dict(n=n, seed=seed, videos=2 * n, frames=int(len(ys)), auc_test=auc_te, auc_val=auc_va,
                         epochs_ran=info.get("epochs"), seconds=info.get("seconds")))
        print(f"     → {tag}: {len(ys)} khung | AUC video test {auc_te:.3f} | val {auc_va:.3f}")

summary = []
for n in sizes:
    te = np.array([r["auc_test"] for r in runs if r["n"] == n])
    va = np.array([r["auc_val"] for r in runs if r["n"] == n])
    summary.append(dict(n=n, frames=int(np.mean([r["frames"] for r in runs if r["n"] == n])),
                        test_mean=float(te.mean()), test_std=float(te.std()),
                        val_mean=float(va.mean()), val_std=float(va.std())))

# ============ ĐỒ THỊ ============
x = [d["n"] for d in summary]
fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.errorbar(x, [d["test_mean"] for d in summary], yerr=[d["test_std"] for d in summary],
            marker="o", capsize=4, lw=1.8, label="test (trung bình ± độ lệch chuẩn)")
ax.plot(x, [d["val_mean"] for d in summary], "--", marker="s", alpha=.75, label="validation")
ax.axhline(0.5, color="grey", lw=.8, ls=":"); ax.text(x[0], 0.505, "đoán mò (0.5)", fontsize=7, color="grey")
ax.set_xlabel("số video huấn luyện mỗi lớp"); ax.set_ylabel("AUC mức video")
ax.set_title(f"Đường học: AUC theo số video huấn luyện ({args.model}, {args.seeds} seed)")
ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x]); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "scaling.png", dpi=130); plt.close(fig)

J = dict(root=str(T.ROOT), model=args.model, color=bool(args.color), epochs=args.epochs, seeds=args.seeds,
         sizes=sizes, frames_per_video=T.K, img=T.IMG, device=T.DEV,
         pool_videos={"real": len(pool[0]), "fake": len(pool[1])},
         val_videos=len(va_set), test_videos=int(len(np.unique(vte))),
         runs=runs, summary=summary, seconds=time.time() - t0)
(OUT / "scaling.json").write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")

# ============ BẢNG ============
print("\n| video/lớp | khung | AUC video test | AUC val | Δ test so với mức trước |")
print("|---|---|---|---|---|")
prev = None
for d in summary:
    delta = "—" if prev is None else f"{d['test_mean'] - prev:+.3f}"
    print(f"| {d['n']} | {d['frames']} | {d['test_mean']:.3f} ± {d['test_std']:.3f} | {d['val_mean']:.3f} | {delta} |")
    prev = d["test_mean"]
if len(summary) >= 2:
    d = summary[-1]["test_mean"] - summary[-2]["test_mean"]
    print(f"\nBước cuối ({summary[-2]['n']} → {summary[-1]['n']} video/lớp) đổi {d:+.3f} AUC:"
          f" {'vẫn còn dốc — thêm dữ liệu vẫn còn lãi' if d > 0.02 else 'đường cong đã phẳng trong sai số — thêm video gần như không giúp nữa' if d > -0.02 else 'GIẢM — nhiễu của tập test nhỏ, hoặc mô hình khớp validation mà không chuyển sang test'}.")
print(f"\nXong trong {time.time()-t0:.0f}s trên {T.DEV}. Xem {OUT}/scaling.png và {OUT}/scaling.json")
