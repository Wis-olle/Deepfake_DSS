"""Trainer — rút gọn từ DeepfakeBench/training/trainer/trainer.py.
Giữ: train_step (forward → get_losses → backward), train_epoch ghi loss + chỉ số lô, test_one_dataset gom prob/label/name
rồi gọi get_test_metrics, save_best theo metric_scoring, save_ckpt.
Bỏ: DDP, SWA, SAM, tensorboard, pickle đặc trưng, test giữa epoch.
Đổi: chọn checkpoint trên VALIDATION (họ chọn trên tập test) và dừng sớm theo `patience`."""
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metrics.utils import get_test_metrics


class Trainer:
    def __init__(self, config, model, optimizer, logger, metric_scoring="video_auc"):
        self.config, self.model, self.optimizer, self.logger = config, model, optimizer, logger
        self.metric_scoring = metric_scoring
        self.device = torch.device("cuda" if config.get("cuda", "auto") != False and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.log_dir = Path(config["out_dir"]) / config["model_name"]; self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best = {"metric": float("-inf"), "epoch": -1, "state": None}
        self.history = []

    def to_device(self, data_dict):
        for k in data_dict:
            if torch.is_tensor(data_dict[k]):
                data_dict[k] = data_dict[k].to(self.device)

    def train_step(self, data_dict):
        predictions = self.model(data_dict)
        losses = self.model.get_losses(data_dict, predictions)
        self.optimizer.zero_grad(); losses["overall"].backward(); self.optimizer.step()
        return losses, predictions

    def train_epoch(self, epoch, train_data_loader, val_data_loader):
        self.model.train(); t0 = time.time(); rec = defaultdict(list)
        for data_dict in train_data_loader:
            self.to_device(data_dict)
            losses, predictions = self.train_step(data_dict)
            for k, v in losses.items():
                rec[k].append(v.item())
            for k, v in self.model.get_train_metrics(data_dict, predictions).items():
                if v is not None:
                    rec[k].append(v)
        avg = {k: float(np.mean(v)) for k, v in rec.items()}
        val = self.test_one_dataset(val_data_loader)
        m = val[self.metric_scoring]; improved = m > self.best["metric"]
        if improved:
            self.best = {"metric": m, "epoch": epoch, "state": {k: v.detach().clone() for k, v in self.model.state_dict().items()}}
            self.save_ckpt()
        self.history.append({"epoch": epoch, **{f"train_{k}": v for k, v in avg.items()},
                             "val_auc": val["auc"], "val_video_auc": val["video_auc"], "seconds": time.time() - t0})
        self.logger.info(f"epoch {epoch:2d}: loss {avg['overall']:.4f} acc {avg.get('acc', 0):.3f} | "
                         f"val AUC khung {val['auc']:.3f} video {val['video_auc']:.3f}{'  *tốt nhất*' if improved else ''} | {time.time() - t0:.0f}s")
        return improved

    @torch.no_grad()
    def test_one_dataset(self, data_loader):
        self.model.eval(); probs, labels, names = [], [], []
        for data_dict in data_loader:
            self.to_device(data_dict)
            predictions = self.model(data_dict, inference=True)
            probs += predictions["prob"].cpu().tolist(); labels += data_dict["label"].cpu().tolist(); names += data_dict["name"]
        return get_test_metrics(np.array(probs), np.array(labels), names)

    def save_ckpt(self):
        torch.save(self.best["state"], self.log_dir / "ckpt_best.pth")

    def load_best(self):
        if self.best["state"] is not None:
            self.model.load_state_dict(self.best["state"])
