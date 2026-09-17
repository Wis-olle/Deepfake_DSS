# Deepfake_DSS — Nhận biết deepfake bằng mạng nơ-ron tích chập trong một hệ trợ giúp quyết định

Mã nguồn của đồ án học phần **Hệ trợ giúp quyết định** (đề tài *Tìm hiểu mạng nơ-ron và ứng dụng trong khai phá dữ liệu*).
Bài toán: với mỗi video có khuôn mặt, hệ thống nên **tự gỡ, tự cho qua, hay chuyển cho người kiểm duyệt**, với xác suất giả
bao nhiêu, dựa vào dấu hiệu gì, và đáng tin tới đâu khi video đến từ nguồn mô hình chưa từng học.

Mạng nơ-ron tích chập (CNN) chỉ là một khối trong hệ thống; phần "trợ giúp quyết định" nằm ở các khối sau nó:
gộp 24 khung hình thành một điểm cho cả video, đặt ngưỡng theo **ma trận chi phí** (để lọt một deepfake đắt gấp 3 lần
gỡ nhầm một video thật), chia ba dải quyết định, và **Grad-CAM** để người kiểm duyệt thấy mạng đang nhìn vào đâu.

Toàn bộ thí nghiệm chạy được trên **CPU của máy tính xách tay trong dưới 10 phút**, không cần GPU.

---

## Kết quả chính

Dữ liệu: **WildDeepfake** (deepfake thu thập từ internet), tập con 142 video đã khớp phân phối độ sáng giữa hai lớp
(50 + 50 video huấn luyện, 21 + 21 video kiểm tra), 24 khung mỗi video, 96 × 96 điểm ảnh. Miền thứ hai để đo ngoài miền:
1.600 ảnh từ bộ Kaggle "Deepfake and Real Images". Một seed; sai số ước lượng của AUC video trên 42 video khoảng ±0,08.

| Mô hình | Học từ | AUC video (test, 42 video) | EER video | Ngoài miền (AUC, 1.600 ảnh) |
|---|---|---|---|---|
| Hồi quy logistic trên 32×32 điểm ảnh xám | thật + deepfake | 0,673 | — | — |
| SmallCNN tự dựng (97.809 tham số) | thật + deepfake | 0,528 | 0,286 | 0,503 |
| ResNet-18 tiền huấn luyện ImageNet | thật + deepfake | **0,782** | 0,286 | 0,511 |
| Self-Blended Images (ResNet-18) | **chỉ ảnh thật**, ảnh giả tự trộn | 0,662 | 0,333 | **0,542** |

Lớp quyết định (ResNet-18, chi phí lọt : gỡ nhầm = 3 : 1, 42 video test):

| Ngưỡng | Bắt được (trong 21 giả) | Gỡ nhầm (trong 21 thật) | Chi phí kỳ vọng / video |
|---|---|---|---|
| 0,50 (mặc định) | 14 | 5 | 0,619 |
| 0,40 (tốt nhất trên validation) | 17 | 6 | 0,429 |
| 0,25 (quy tắc Bayes của Elkan, τ* = 1 / (1 + 3)) | 20 | 14 | **0,405** |

Bốn điều đáng nhớ:

1. **Con số đẹp đầu tiên là lối tắt.** Trên tập con tải ngẫu nhiên lúc đầu, ResNet-18 đạt AUC video 0,938 nhưng chỉ 0,486
   ngoài miền. Video thật trong tập đó tối hơn video giả một cách hệ thống (độ sáng trung bình 81 so với 113); một hồi quy
   logistic chỉ nhìn 1.024 điểm ảnh xám đã đạt 0,729, và Grad-CAM cho thấy với video thật mạng nhìn vào nền tối chứ không
   nhìn vào mặt. Sau khi chọn lại dữ liệu cho hai lớp cùng phân phối độ sáng, "máy dò lối tắt" (hồi quy logistic chỉ thấy
   một con số độ sáng) về 0,52 ≈ 0,5 và ResNet-18 còn 0,78–0,79: đó mới là con số thật.
2. **Mạng tự dựng không vượt được hồi quy logistic** trên vài chục video; học chuyển giao từ ImageNet thắng rõ. Thêm video
   trong khoảng 8–32 video mỗi lớp không làm SmallCNN tốt lên trên tập test.
3. **Ngoài miền mọi mô hình học trên deepfake của WildDeepfake đều về đoán mò** (0,50–0,51), đúng như DF40 và
   Deepfake-Eval-2024 đo ở quy mô lớn. Mô hình duy nhất nhích lên là Self-Blended Images, mô hình chưa từng thấy một
   deepfake nào khi học.
4. **Ngưỡng là quyết định của chính sách, không phải của mô hình.** Cùng một mô hình, đổi tỉ lệ chi phí là đổi ngưỡng và
   đổi số video bị gỡ nhầm; không cần huấn luyện lại.

---

## Cấu trúc kho

```
deepfake/
├── get_wdf.py               tải kho ứng viên WildDeepfake từ HuggingFace, đo độ sáng từng video, chọn tập con khớp → wdf/
├── train_deepfake_cnn.py    kịch bản một tệp, 9 bước: thống kê dữ liệu → chia theo video → baseline + máy dò lối tắt
│                            → SmallCNN (có / không tăng cường) → ResNet-18 → ngoài miền → ngưỡng theo chi phí → Grad-CAM
├── data_scaling.py          đường học: AUC theo số video huấn luyện
├── export_test_subset.py    xuất 24 khung/video của 42 video test (57 MB) để chia sẻ nội bộ
└── dfb/                     dự án dựng lại theo khung DeepfakeBench (xem dfb/README.md)
    ├── config/              train_config.yaml + detector/{small_cnn,resnet18,sbi}.yaml: mọi siêu tham số nằm ở đây
    ├── dataset/             nạp WildDeepfake theo video, tăng cường, chia validation theo cụm màu; Self-Blended Images
    ├── detectors/           hợp đồng chung (base_detector.py) + ba bộ phát hiện, đăng ký theo tên
    ├── metrics/             ACC, AUC khung, EER, AP, AUC video (gộp khung theo video)
    ├── trainer/             vòng huấn luyện, dừng sớm, chọn checkpoint trên validation
    ├── train.py / test.py   điểm vào
    └── test_dfb.py          4 kiểm tra tối thiểu, 30 giây, không cần dữ liệu
requirements.txt
```

Dữ liệu, kết quả chạy và checkpoint không nằm trong kho (xem mục *Dữ liệu và giấy phép*).

---

## Chạy lại

### 1. Môi trường

Cần Python 3.10 hoặc 3.11 **64-bit**. Trên Windows, nếu máy có nhiều bản Python, gọi đích danh (`py -3.10`) để tránh
trỏ nhầm sang một bản không có bánh xe PyTorch. Bánh xe `torch` trên PyPI cho Windows đã là bản CPU.

```bash
py -3.10 -m venv venv            # Linux/macOS: python3.10 -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
```

### 2. Dữ liệu

```bash
cd deepfake
python get_wdf.py                # ~25 phút: tải kho ứng viên (~2 GB) + 1 shard miền thứ hai, chọn tập con khớp → wdf/
python get_wdf.py --no-download  # chỉ chọn lại tập con từ kho đã có
```

`wdf/` sau đó gồm `real_train/`, `fake_train/`, `real_test/`, `fake_test/` (mỗi video một thư mục khung mặt đã cắt) và
`selection.json` ghi danh sách video được chọn cùng độ sáng trước / sau khi khớp.

### 3. Kịch bản một tệp (khoảng 5 phút CPU)

```bash
python train_deepfake_cnn.py
```

In ra từng bước và ghi vào `out/`: `deepfake_results.json` và `.md` (mọi con số), `curves_cnn.png` (đường huấn luyện),
`wdf_grid.png` (lưới dữ liệu), `bright_hist.png` (phân phối độ sáng), `worst.png` (mười khung sai nặng nhất),
`gradcam.png` (mạng nhìn vào đâu). Tuỳ chọn: `python data_scaling.py --sizes 8,16,32 --seeds 2` cho đường học.

### 4. Gói theo khung DeepfakeBench

```bash
cd dfb
python test_dfb.py                                              # 4 dòng OK trong 30 giây
python train.py --detector_path config/detector/small_cnn.yaml  # ~1,5 phút
python train.py --detector_path config/detector/resnet18.yaml   # ~2 phút
python train.py --detector_path config/detector/sbi.yaml        # ~4 phút; mở ../out/dfb/sbi/sbi_examples.png để xem ảnh giả tự trộn
```

Mỗi lần chạy ghi vào `out/dfb/<detector>/`: `training.log`, `metrics.json` (lịch sử từng epoch, kết quả test trong miền và
ngoài miền, ba ngưỡng quyết định, xác suất từng video), `results.md`, `ckpt_best.pth`. Muốn thí nghiệm khác, chép một
tệp YAML trong `config/detector/` và đổi vài dòng; không phải sửa mã. Đổi seed bằng `--manualSeed`.

---

## Phương pháp, tóm tắt

- **Đơn vị thống kê là video, không phải khung.** Chia học / validation / test theo ranh giới video; điểm video là trung bình
  xác suất của 24 khung; thước đo chính là AUC ở mức video, báo cáo kèm AUC khung, EER, AP.
- **Kiểm tra trước khi tin.** Loss lúc khởi tạo phải bằng ln 2; mạng phải học thuộc được 32 ảnh; hai baseline hồi quy
  logistic (điểm ảnh, phổ tần số) đặt mốc; máy dò lối tắt (hồi quy logistic trên 1 số độ sáng) phải về 0,5.
- **Validation tách theo "nguồn" gần đúng.** Không có nhãn nguồn, gom 100 video thành 5 cụm theo màu trung bình bằng
  KMeans và lấy trọn một cụm làm validation, để validation không dùng chung lối tắt với tập học.
- **Học chuyển giao và Self-Blended Images.** ResNet-18 của torchvision với trọng số ImageNet, thay lớp cuối, tinh chỉnh
  toàn mạng với learning rate 1e-4. SBI cài lại theo mã gốc của Shiohara với cùng tham số biến đổi màu, dịch, co giãn và
  trộn, nhưng mặt nạ là elip hoặc đa giác lồi quanh tâm ảnh thay cho mặt nạ từ điểm mốc (WildDeepfake chỉ cho ảnh mặt đã cắt).
- **Lớp quyết định.** Ngưỡng τ* = C_gỡ nhầm / (C_gỡ nhầm + C_lọt) theo Elkan (2001); ba dải (dưới 0,1 tự cho qua, trên 0,7
  tự gỡ, giữa chuyển người kèm Grad-CAM); phân tích độ nhạy theo tỉ lệ chi phí.
- **Học theo DeepfakeBench, trừ hai chỗ.** Gói `dfb/` dùng cùng bố cục, cùng hợp đồng bộ phát hiện, cùng bộ chỉ số và danh
  sách tăng cường của DeepfakeBench. Hai chỗ cố ý làm khác: checkpoint được chọn trên **validation** (mã gốc chọn trên tập
  test); khung được gộp theo cặp `phần/tên video` vì WildDeepfake đặt tên video trùng nhau giữa `real_test/` và `fake_test/`.

---

## Dữ liệu và giấy phép

- **WildDeepfake** (Zi và cộng sự, ACM Multimedia 2020; bản mirror `huggingface.co/datasets/xingjunm/WildDeepfake`) chỉ cấp
  cho mục đích nghiên cứu. Kho này **không phân phối lại** bất kỳ khung hình nào; hãy tải bằng `get_wdf.py` và tuân theo
  điều khoản của bộ dữ liệu. Đây là khuôn mặt người thật, một số bị ghép vào nội dung họ không đồng ý.
- **Miền thứ hai**: bộ Kaggle "Deepfake and Real Images" (bản mirror `huggingface.co/datasets/Hemg/deepfake-and-real-images`);
  thẻ dữ liệu không ghi kỹ thuật sinh ảnh giả, nên chỉ dùng để đo ngoài miền.
- Trọng số ResNet-18 lấy từ torchvision (giấy phép BSD). Mã DeepfakeBench và SelfBlendedImages được đọc để học cấu trúc và
  viết lại, không sao chép tệp.

## Hạn chế

Một seed cho bảng kết quả chính; 42 video test nên sai số AUC video khoảng ±0,08 (một độ lệch chuẩn); ảnh 96 px và 24 khung
mỗi video vì ngân sách CPU; một miền thứ hai duy nhất; SBI là bản rút gọn (96 px, mặt nạ elip, 39 video thật) nên con số của
nó là sàn chứ không phải trần của phương pháp; xác suất chưa được hiệu chuẩn; không có nhãn danh tính để bảo đảm tách theo người;
không có video Việt Nam nên không kết luận gì cho nội dung tiếng Việt. Việc đáng làm tiếp theo, theo thứ tự: chạy 5 seed;
đẩy SBI lên 128–160 px với đủ video thật trong kho; hiệu chuẩn bằng temperature scaling; đo trên miền thứ ba;
FaceForensics++ khi được cấp quyền.

## Tài liệu chính

- B. Zi và cộng sự, "WildDeepfake: A challenging real-world dataset for deepfake detection," ACM MM 2020.
- Z. Yan và cộng sự, "DeepfakeBench: A comprehensive benchmark of deepfake detection," NeurIPS 2023 — [github.com/SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).
- K. Shiohara, T. Yamasaki, "Detecting deepfakes with self-blended images," CVPR 2022 — [github.com/mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages).
- Z. Yan và cộng sự, "DF40: Toward next-generation deepfake detection," NeurIPS 2024; N. A. Chandra và cộng sự, "Deepfake-Eval-2024," arXiv:2503.02857.
- R. Geirhos và cộng sự, "Shortcut learning in deep neural networks," Nature Machine Intelligence, 2020.
- C. Elkan, "The foundations of cost-sensitive learning," IJCAI 2001.
- R. R. Selvaraju và cộng sự, "Grad-CAM," ICCV 2017.
- A. Karpathy, "A recipe for training neural networks," 2019 — quy trình mà kịch bản một tệp bám theo.
