# dfb — dự án deepfake dựng lại theo khung DeepfakeBench

Học theo [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) (Yan et al., NeurIPS 2023; mã nguồn tại
`nguon-tai-lieu/DeepfakeBench/`, ghi chú đọc mã ở `nguon-tai-lieu/DeepfakeBench-ghi-chu.md`).
Cùng bố cục, cùng "hợp đồng" giữa các phần, chỉ bỏ những gì máy CPU và dữ liệu WildDeepfake không có.

```
dfb/
├── config/
│   ├── train_config.yaml          # đường dẫn dữ liệu, nhãn, cách chia validation, tập test, ma trận chi phí
│   └── detector/{small_cnn,resnet18,sbi}.yaml     # mỗi detector một YAML (giống config/detector/*.yaml của họ)
├── dataset/
│   ├── wdf_dataset.py             # WDFDataset ← DeepfakeAbstractBaseDataset; + chia validation theo video
│   ├── kaggle_dataset.py          # miền thứ hai (kiểm tra chéo bộ dữ liệu)
│   ├── sbi_api.py                 # Self-Blended Images ← sbi_api.py của Shiohara; mặt nạ elip thay landmark
│   └── sbi_dataset.py             # SBIDataset: chỉ ảnh thật → cặp (giả tự trộn, thật)
├── detectors/
│   ├── __init__.py                # Registry + DETECTOR + LOSSFUNC
│   ├── base_detector.py           # AbstractDetector, 7 phương thức trừu tượng (chép nguyên)
│   ├── small_cnn_detector.py      # @DETECTOR.register_module("small_cnn")
│   ├── resnet18_detector.py       # @DETECTOR.register_module("resnet18")
│   └── sbi_detector.py            # @DETECTOR.register_module("sbi") = ResNet-18, khác ở DỮ LIỆU
├── metrics/utils.py               # ACC, AUC khung, EER, AP, AUC video (get_test_metrics)
├── trainer/trainer.py             # train_step / train_epoch / test_one_dataset / save_ckpt
├── train.py                       # YAML → dữ liệu → DETECTOR[model_name] → optimizer → Trainer → test
├── test.py                        # đánh giá lại một ckpt_best.pth
└── test_dfb.py                    # 4 kiểm tra tối thiểu (ln2, overfit, SBI, gom video)
```

## Chạy

```bash
cd deepfake/dfb
..\..\venv\Scripts\python.exe test_dfb.py                                        # < 30 s, không cần dữ liệu
..\..\venv\Scripts\python.exe train.py --detector_path config/detector/small_cnn.yaml
..\..\venv\Scripts\python.exe train.py --detector_path config/detector/resnet18.yaml
..\..\venv\Scripts\python.exe train.py --detector_path config/detector/sbi.yaml
..\..\venv\Scripts\python.exe test.py  --detector_path config/detector/resnet18.yaml --weights_path ../out/dfb/resnet18/ckpt_best.pth
```

Kết quả mỗi detector ở `deepfake/out/dfb/<model_name>/`: `training.log`, `metrics.json` (lịch sử epoch, chỉ số test,
ngưỡng quyết định, xác suất từng video), `ckpt_best.pth`, `results.md`, và `sbi_examples.png` với SBI.
Tuỳ chọn: `--rgb_dir ../wdf_v1` (tập cũ lệch nguồn), `--val_split random`, `--nEpochs N`, `--manualSeed S`.

## Họ có gì, ta lấy gì

| DeepfakeBench | dfb | Vì sao |
|---|---|---|
| `AbstractDetector` với features / classifier / forward / build_backbone / build_loss / get_losses / get_train_metrics | chép nguyên | thêm detector = thêm một tệp, `train.py` không đổi |
| `@DETECTOR.register_module` + `Registry` | chép nguyên | cùng lý do |
| `data_dict{'image','label'}` → `pred_dict{'cls','prob','feat'}`, 2 lớp + softmax + cross-entropy | giữ | trainer không cần biết mô hình bên trong |
| `DeepfakeAbstractBaseDataset`: gom theo video, `frame_num`, `data_aug`, `collate_fn` | giữ hình dạng; ảnh nạp vào RAM | 3 nghìn khung 96 px ≈ 80 MB |
| tăng cường: flip .5, rotate ±10° .5, blur .5, sáng/tương phản ±0.1 .5, JPEG 40–100 .5 (albumentations) | cùng danh sách, viết bằng PIL | không thêm thư viện |
| `get_test_metrics`: ACC, AUC, EER, AP, AUC video | giữ; gom theo `split/video` thay vì `parts[-2]` | WildDeepfake đặt tên video trùng giữa real_test/ và fake_test/ |
| `sbi_api.py`: mask từ 81 landmark, source transforms, randaffine, dynamic blend | cùng 4 bước, cùng tham số; mask = elip / đa giác quanh tâm ảnh | ảnh mặt đã cắt sẵn, không có landmark |
| `sbi_dataset.py`: chỉ ảnh thật, trả cặp, collate nối | giữ | mô hình chưa từng thấy deepfake thật |
| `Trainer`: train_step, train_epoch, test_one_dataset, save_best | giữ; **chọn checkpoint trên validation** | họ chọn checkpoint ngay trên tập test: rò rỉ lựa chọn mô hình |
| YAML detector + YAML chung, `train.py` gộp hai tệp | giữ | mọi con số ở một chỗ |
| 13 bộ dữ liệu FF++/Celeb-DF/DFDC, tiền xử lý dlib + lmdb | bỏ | chỉ có WildDeepfake (mặt cắt sẵn) + mirror Kaggle |
| DDP, SWA, SAM, tensorboard, 20 detector, 4 backbone | bỏ | CPU, một máy |
| không có bước ra quyết định | thêm ngưỡng mức video theo chi phí (Elkan) | đây là bài DSS |

## Kết quả (14/09/2026; một seed, 96 px, CPU; `wdf/` = 50+50 train, 21+21 test; validation = cụm màu 21 video)

| Detector | Huấn luyện thấy gì | val AUC video | test AUC khung | test AUC video | Kaggle AUC (ngoài miền) | Thời gian |
|---|---|---|---|---|---|---|
| `small_cnn` | thật + deepfake | 0,718 | 0,526 | 0,528 | 0,503 | 91 s |
| `resnet18` | thật + deepfake | 0,945 | 0,689 | 0,782 | 0,511 | 126 s |
| `sbi` | chỉ ảnh thật | 0,667 | 0,603 | 0,662 | 0,542 | 248 s |

Diễn giải và ngưỡng quyết định: `neural/README.md`, mục "Học theo DeepfakeBench". Số chi tiết: `out/dfb/<model_name>/metrics.json`.
