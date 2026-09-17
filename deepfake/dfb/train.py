"""Điểm vào huấn luyện — học theo DeepfakeBench/training/train.py:
YAML detector + train_config → hạt giống → dữ liệu (plain | blend) → DETECTOR[model_name] → optimizer → Trainer → vòng epoch → test.
Khác họ: validation theo video + dừng sớm (họ chọn checkpoint trên tập test — sai phương pháp); không DDP / lmdb / tensorboard;
thêm bước ngưỡng quyết định ở mức video (chi phí Elkan) vì đây là bài DSS.

Chạy (trong thư mục dfb/):
  ..\\..\\venv\\Scripts\\python.exe train.py --detector_path config/detector/small_cnn.yaml
  ..\\..\\venv\\Scripts\\python.exe train.py --detector_path config/detector/resnet18.yaml
  ..\\..\\venv\\Scripts\\python.exe train.py --detector_path config/detector/sbi.yaml
Kết quả: ../out/dfb/<model_name>/{training.log, metrics.json, ckpt_best.pth, results.md[, sbi_examples.png]}"""
import argparse, json, logging, pathlib, random, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch, yaml
from torch.utils.data import DataLoader

from dataset.wdf_dataset import WDFDataset, split_train_val
from dataset.sbi_dataset import SBIDataset
from dataset.kaggle_dataset import KaggleDataset
from detectors import DETECTOR
from trainer.trainer import Trainer

HERE = pathlib.Path(__file__).resolve().parent


def load_config(detector_path, overrides):
    config = yaml.safe_load(open(detector_path, encoding="utf-8"))
    config.update(yaml.safe_load(open(HERE / "config" / "train_config.yaml", encoding="utf-8")))
    config.update({k: v for k, v in overrides.items() if v is not None})
    for k in ("rgb_dir", "out_dir"):
        config[k] = str((HERE / config[k]).resolve()) if not pathlib.Path(config[k]).is_absolute() else config[k]
    return config


def init_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def create_logger(path):
    logger = logging.getLogger("dfb"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")); logger.addHandler(h)
    return logger


def choose_optimizer(model, config):
    o = config["optimizer"]; t = o["type"]
    if t == "adam":
        return torch.optim.Adam(model.parameters(), lr=o["adam"]["lr"], betas=(o["adam"]["beta1"], o["adam"]["beta2"]),
                                eps=o["adam"]["eps"], weight_decay=o["adam"]["weight_decay"], amsgrad=o["adam"]["amsgrad"])
    if t == "sgd":
        return torch.optim.SGD(model.parameters(), lr=o["sgd"]["lr"], momentum=o["sgd"]["momentum"], weight_decay=o["sgd"]["weight_decay"])
    raise NotImplementedError(t)


def save_sbi_examples(ds, path, n=8):
    """NHÌN dữ liệu tự sinh trước khi huấn luyện: hàng 1 thật, hàng 2 giả tự trộn, hàng 3 mặt nạ trộn."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, n, figsize=(1.6 * n, 5))
    for c in range(n):
        f, r, m = ds.sbi(ds.image_list[c * len(ds) // n])
        for row, im, cmap in ((0, r, None), (1, f, None), (2, m, "gray")):
            ax[row, c].imshow(im, cmap=cmap, vmin=0, vmax=1 if cmap else None); ax[row, c].axis("off")
    for row, t in enumerate(("thật", "giả tự trộn (SBI)", "mặt nạ trộn")):
        ax[row, 0].set_title(t, fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def prepare_training_data(config, logger, out):
    full = WDFDataset(config, mode="train")
    train_set, val_set, info = split_train_val(full, config["val_frac"], config["manualSeed"], config["val_split"])
    logger.info(f"train {len(train_set.videos())} video / {len(train_set)} khung | validation = {info}, {len(val_set)} khung")
    if config.get("dataset_type") == "blend":
        train_set = SBIDataset(config, base=train_set)
        logger.info(f"SBI: chỉ dùng {len(train_set)} khung THẬT từ {len(train_set.videos())} video; mỗi khung sinh một cặp (thật, giả tự trộn)")
        save_sbi_examples(train_set, out / "sbi_examples.png"); logger.info(f"ví dụ SBI: {out / 'sbi_examples.png'}")
    train_loader = DataLoader(train_set, batch_size=config["train_batchSize"], shuffle=True, num_workers=config["workers"], collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set, batch_size=config["test_batchSize"], shuffle=False, num_workers=config["workers"], collate_fn=val_set.collate_fn)
    return train_loader, val_loader


def prepare_testing_data(config, logger):
    loaders = {}
    for name in config["test_dataset"]:
        try:
            ds = WDFDataset(config, mode="test") if name == "wdf" else KaggleDataset(config, config["kaggle_per_class"])
        except Exception as e:
            logger.info(f"bỏ qua tập test {name}: {e}"); continue
        loaders[name] = DataLoader(ds, batch_size=config["test_batchSize"], shuffle=False, num_workers=config["workers"], collate_fn=ds.collate_fn)
        logger.info(f"test {name}: {len(ds.videos())} video / {len(ds)} khung")
    return loaders


def decision_report(val, test, cost, logger):
    """Bước DSS: chọn ngưỡng ở mức VIDEO trên validation, áp lên test; so với 0.5 và ngưỡng Elkan C_FP/(C_FP+C_FN)."""
    def cost_at(y, p, t):
        pred = p >= t
        return (cost["miss"] * ((~pred) & (y == 1)).sum() + cost["false_alarm"] * (pred & (y == 0)).sum()) / len(y)
    yv, pv, yt, pt = val["video_label"], val["video_pred"], test["video_label"], test["video_pred"]
    grid = np.linspace(0.05, 0.95, 91); thr = float(grid[np.argmin([cost_at(yv, pv, t) for t in grid])])
    elkan = cost["false_alarm"] / (cost["false_alarm"] + cost["miss"]); rows = []
    for name, t in (("0.50", 0.5), (f"tốt nhất trên val={thr:.2f}", thr), (f"Elkan={elkan:.2f}", elkan)):
        pred = pt >= t
        rows.append(dict(name=name, thr=float(t), tp=int((pred & (yt == 1)).sum()), P=int((yt == 1).sum()),
                         fp=int((pred & (yt == 0)).sum()), N=int((yt == 0).sum()), cost=float(cost_at(yt, pt, t))))
        logger.info(f"  ngưỡng {name:24s}: bắt {rows[-1]['tp']}/{rows[-1]['P']} video giả, gỡ nhầm {rows[-1]['fp']}/{rows[-1]['N']} video thật, chi phí {rows[-1]['cost']:.3f}/video")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Huấn luyện một detector theo khung DeepfakeBench")
    ap.add_argument("--detector_path", required=True, help="YAML trong config/detector/")
    ap.add_argument("--test_dataset", nargs="+", default=None)
    ap.add_argument("--rgb_dir", default=None); ap.add_argument("--out_dir", default=None)
    ap.add_argument("--nEpochs", type=int, default=None); ap.add_argument("--manualSeed", type=int, default=None)
    ap.add_argument("--val_split", default=None, choices=["cluster", "random"])
    args = ap.parse_args()
    config = load_config(args.detector_path, vars(args))
    out = pathlib.Path(config["out_dir"]) / config["model_name"]; out.mkdir(parents=True, exist_ok=True)
    logger = create_logger(out / "training.log")
    logger.info(f"detector {config['model_name']} | dữ liệu {config['rgb_dir']} | kết quả {out}")
    logger.info("cấu hình: " + json.dumps({k: v for k, v in config.items() if k not in ("data_aug", "optimizer", "label_dict")}, ensure_ascii=False))
    init_seed(config["manualSeed"]); t0 = time.time()

    train_loader, val_loader = prepare_training_data(config, logger, out)
    test_loaders = prepare_testing_data(config, logger)
    model = DETECTOR[config["model_name"]](config)
    logger.info(f"tham số: {sum(p.numel() for p in model.parameters()):,}")
    trainer = Trainer(config, model, choose_optimizer(model, config), logger, config["metric_scoring"])

    bad = 0
    for epoch in range(config["nEpochs"]):
        improved = trainer.train_epoch(epoch, train_loader, val_loader)
        bad = 0 if improved else bad + 1
        if bad >= config.get("patience", 10 ** 9):
            logger.info(f"dừng sớm: {bad} epoch không cải thiện"); break
    trainer.load_best()
    logger.info(f"checkpoint tốt nhất: epoch {trainer.best['epoch']}, val {config['metric_scoring']} {trainer.best['metric']:.3f}")

    results = {}; val = trainer.test_one_dataset(val_loader)
    for name, loader in test_loaders.items():
        m = trainer.test_one_dataset(loader)
        results[name] = {k: m[k] for k in ("acc", "auc", "eer", "ap", "video_auc", "video_eer")}
        logger.info(f"test {name:7s}: ACC {m['acc']:.3f} | AUC khung {m['auc']:.3f} | EER {m['eer']:.3f} | AP {m['ap']:.3f} | AUC video {m['video_auc']:.3f}")
        if name == "wdf":
            logger.info(f"ngưỡng quyết định mức video (chi phí lọt : gỡ nhầm = {config['cost']['miss']:.0f} : {config['cost']['false_alarm']:.0f})")
            results[name]["thresholds"] = decision_report(val, m, config["cost"], logger)
            results[name]["video_pred"] = {v: float(p) for v, p in zip(m["video_names"], m["video_pred"])}
    J = dict(model_name=config["model_name"], detector_path=args.detector_path, rgb_dir=config["rgb_dir"], seed=config["manualSeed"],
             params=sum(p.numel() for p in model.parameters()), best_epoch=trainer.best["epoch"], best_val=trainer.best["metric"],
             history=trainer.history, test=results, seconds=time.time() - t0)
    (out / "metrics.json").write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["| Detector | val AUC video | " + " | ".join(f"{n} AUC khung | {n} AUC video" for n in results) + " |",
          "|---|---|" + "---|---|" * len(results),
          f"| {config['model_name']} | {trainer.best['metric']:.3f} | " + " | ".join(f"{r['auc']:.3f} | {r['video_auc']:.3f}" for r in results.values()) + " |"]
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("\n" + "\n".join(md)); logger.info(f"xong trong {time.time() - t0:.0f}s trên {trainer.device}")


if __name__ == "__main__":
    main()
