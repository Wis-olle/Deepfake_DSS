"""Đánh giá một checkpoint đã lưu — học theo DeepfakeBench/training/test.py.
Chạy (trong dfb/):  ..\\..\\venv\\Scripts\\python.exe test.py --detector_path config/detector/resnet18.yaml --weights_path ../out/dfb/resnet18/ckpt_best.pth --test_dataset wdf kaggle"""
import argparse, logging, pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")
import torch

from detectors import DETECTOR
from trainer.trainer import Trainer
from train import load_config, prepare_testing_data, init_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_path", required=True); ap.add_argument("--weights_path", required=True)
    ap.add_argument("--test_dataset", nargs="+", default=None); ap.add_argument("--rgb_dir", default=None)
    args = ap.parse_args()
    config = load_config(args.detector_path, {"test_dataset": args.test_dataset, "rgb_dir": args.rgb_dir})
    logger = logging.getLogger("dfb"); logger.handlers.clear(); logger.setLevel(logging.INFO); logger.addHandler(logging.StreamHandler(sys.stdout))
    init_seed(config["manualSeed"])
    model = DETECTOR[config["model_name"]](config)
    model.load_state_dict(torch.load(args.weights_path, map_location="cpu")); logger.info(f"nạp {args.weights_path}")
    trainer = Trainer(config, model, None, logger, config["metric_scoring"])
    for name, loader in prepare_testing_data(config, logger).items():
        m = trainer.test_one_dataset(loader)
        logger.info(f"test {name:7s}: ACC {m['acc']:.3f} | AUC khung {m['auc']:.3f} | EER {m['eer']:.3f} | AP {m['ap']:.3f} | AUC video {m['video_auc']:.3f}")


if __name__ == "__main__":
    main()
