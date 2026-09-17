"""Đo lường — rút gọn từ DeepfakeBench/training/metrics/utils.py (get_test_metrics) và base_metrics_class.py
(calculate_metrics_for_train). Cùng bộ chỉ số: ACC, AUC khung, EER, AP, AUC video (trung bình xác suất các khung).
Khác MỘT chỗ: gom khung theo 'split/video' thay vì chỉ tên video (họ lấy parts[-2]) — WildDeepfake đặt tên video
trùng nhau giữa real_test/ và fake_test/ (vd. 101/), lấy parts[-2] sẽ trộn video thật với video giả cùng tên."""
import numpy as np
import torch
from sklearn import metrics


def _eer(fpr, tpr):
    fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


def calculate_metrics_for_train(label, output):
    """Chỉ số trên một lô: (auc, eer, acc, ap). auc/eer/ap = None nếu lô chỉ có một lớp."""
    prob = torch.softmax(output, dim=1)[:, 1] if output.dim() == 2 and output.size(1) == 2 else output
    acc = (output.argmax(1) == label).float().mean().item()
    y, p = label.cpu().numpy(), prob.detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return None, None, acc, None
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    return float(metrics.auc(fpr, tpr)), _eer(fpr, tpr), acc, float(metrics.average_precision_score(y, p))


def video_id(name):
    """'real_test/101/7' → 'real_test/101'  (chấp nhận cả dấu \\ của Windows như mã gốc)."""
    return "/".join(name.replace("\\", "/").split("/")[:-1])


def get_video_metrics(names, pred, label):
    d = {}
    for n, p, l in zip(names, pred, label):
        d.setdefault(video_id(n), []).append((float(p), int(l)))
    vp = np.array([np.mean([x[0] for x in v]) for v in d.values()])
    vl = np.array([int(round(np.mean([x[1] for x in v]))) for v in d.values()])
    fpr, tpr, _ = metrics.roc_curve(vl, vp, pos_label=1)
    return float(metrics.auc(fpr, tpr)), _eer(fpr, tpr), vp, vl, list(d)


def get_test_metrics(y_pred, y_true, img_names):
    y_pred = np.asarray(y_pred, dtype=float).squeeze()
    y_true = np.clip(np.asarray(y_true).astype(int), 0, 1)          # họ: y_true[y_true>=1]=1 (nhãn UCF nhiều loại giả)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    acc = float(((y_pred > 0.5).astype(int) == y_true).mean())
    v_auc, v_eer, vp, vl, vn = get_video_metrics(img_names, y_pred, y_true)
    return {"acc": acc, "auc": float(metrics.auc(fpr, tpr)), "eer": _eer(fpr, tpr),
            "ap": float(metrics.average_precision_score(y_true, y_pred)),
            "video_auc": v_auc, "video_eer": v_eer,
            "pred": y_pred, "label": y_true, "video_pred": vp, "video_label": vl, "video_names": vn}
