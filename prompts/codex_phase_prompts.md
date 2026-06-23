# Codex Phase Prompts

File này chứa các prompt nhỏ để dùng từng giai đoạn. Dùng khi không muốn dán master prompt quá dài.

---

## Prompt Phase 0 — Initialize new repo

```text
Bạn là AI Research Engineer. Hãy khởi tạo skeleton cho một repo nghiên cứu mới tên "xai-evidence-small-object-detection".

Mục tiêu đề tài: XAI-guided Evidence Preservation for Small Object Detection.

Tạo cấu trúc thư mục chuẩn:
- configs/
- docs/
- prompts/
- src/xai_evidence_sod/
- scripts/
- tests/
- notebooks/
- experiments/
- artifacts/
- paper/

Tạo README.md, requirements.txt, pyproject.toml tối thiểu, __init__.py, .gitkeep cho thư mục rỗng.

Không viết training code ở pha này. Chỉ tạo nền repo sạch và giải thích ngắn từng thư mục.

Sau khi hoàn thành, báo cáo file đã tạo, cách kiểm tra, và phase tiếp theo.
```

---

## Prompt Phase 1 — Research docs

```text
Hãy tạo bộ tài liệu nghiên cứu nền cho repo mới.

Tạo/cập nhật các file:
- docs/00_research_manifest.md
- docs/01_problem_statement.md
- docs/02_research_gap_and_hypotheses.md
- docs/03_methodology.md
- docs/04_experimental_protocol.md
- docs/05_metrics_and_ablation.md
- docs/06_implementation_roadmap.md
- docs/experiment_log.md

Nội dung phải xoay quanh ý tưởng: dùng XAI/saliency để đo evidence của small objects qua P2/P3/P4, phát hiện evidence drop, và sau đó dùng Evidence Preservation Loss để giảm drop.

Không viết code trong phase này. Ưu tiên tính khoa học, câu hỏi nghiên cứu, giả thuyết, metric, protocol và tiêu chí nghiệm thu.
```

---

## Prompt Phase 2 — Dataset inspection

```text
Hãy triển khai phase dataset inspection cho repo mới.

Tạo:
- src/xai_evidence_sod/data/yolo_dataset.py
- src/xai_evidence_sod/data/size_analysis.py
- scripts/inspect_dataset.py
- configs/dataset/example_yolo.yaml

Yêu cầu inspect_dataset.py:
- đọc data.yaml YOLO
- parse labels
- tính bbox area theo pixel với imgsz tùy chọn
- phân nhóm tiny/small/medium/large
- thống kê số ảnh, số object, class distribution, bbox area distribution
- lưu artifacts/dataset_inspection/dataset_summary.md
- lưu artifacts/dataset_inspection/bbox_stats.csv
- có argparse: --data, --imgsz, --out, --max-images

Không train model. Không thêm XAI.
```

---

## Prompt Phase 3 — Baseline training

```text
Hãy triển khai baseline training/evaluation tối giản cho repo mới.

Ưu tiên dùng Ultralytics YOLO wrapper để giảm rủi ro.

Tạo:
- src/xai_evidence_sod/models/yolo_wrapper.py
- src/xai_evidence_sod/utils/config.py
- src/xai_evidence_sod/utils/seed.py
- scripts/train_baseline.py
- scripts/eval_baseline.py
- configs/train/baseline_yolo.yaml

Yêu cầu:
- train_baseline.py đọc config yaml
- lưu output vào experiments/<run_name>/
- copy config vào run folder
- eval_baseline.py đọc checkpoint và lưu metrics.json + summary.md
- baseline không phụ thuộc bất kỳ module XAI/evidence nào

Sau khi hoàn thành, cung cấp command chạy mẫu.
```

---

## Prompt Phase 4 — Feature hooks

```text
Hãy triển khai hệ thống inspect layer và hook feature maps.

Tạo:
- src/xai_evidence_sod/xai/layer_registry.py
- src/xai_evidence_sod/xai/feature_hooks.py
- scripts/inspect_model_layers.py
- scripts/debug_feature_hooks.py

Yêu cầu:
- inspect_model_layers.py in index/name/type của các layer
- layer_registry cho phép cấu hình map P2/P3/P4 sang layer name/index
- feature_hooks.py register forward hooks và capture tensor
- debug_feature_hooks.py load model + 1 batch, forward, in shape P2/P3/P4, kiểm tra NaN/Inf
- không thay đổi output detector

Không thêm saliency/loss ở phase này.
```

---

## Prompt Phase 5 — Saliency

```text
Hãy triển khai saliency generation nhẹ cho feature maps.

Tạo:
- src/xai_evidence_sod/xai/saliency.py
- scripts/debug_saliency_one_batch.py

Implement methods:
- activation_mean
- activation_abs_mean
- eigen_cam_like nếu không quá phức tạp

Input feature map: [B, C, H, W]
Output saliency: [B, 1, H, W], normalize [0, 1]

Script debug:
- load 1 batch
- hook P2/P3/P4
- sinh saliency
- lưu ảnh visualization vào artifacts/xai_debug/

Không thêm evidence loss.
```

---

## Prompt Phase 6 — Evidence metrics

```text
Hãy triển khai evidence metrics.

Tạo:
- src/xai_evidence_sod/evidence/masks.py
- src/xai_evidence_sod/evidence/metrics.py
- scripts/debug_evidence_metrics.py

Implement:
- scale_bbox_to_feature
- make_bbox_mask
- compute_inside_energy
- compute_outside_energy
- compute_evidence_ratio
- compute_evidence_drop
- summarize_evidence_batch

Yêu cầu xử lý:
- bbox tọa độ ảnh gốc
- feature/saliency khác resolution ảnh
- bbox quá nhỏ
- clamp out-of-bound
- eps tránh chia 0

Debug script dùng tensor giả để test logic trước khi dùng model thật.
```

---

## Prompt Phase 7 — Baseline evidence analysis

```text
Hãy triển khai baseline evidence analysis.

Chỉ làm phase này nếu baseline training/eval, feature hooks, saliency và evidence metrics đã chạy được.

Tạo:
- scripts/analyze_baseline_evidence.py

Script phải:
- load baseline checkpoint
- chạy trên validation subset
- hook P2/P3/P4
- sinh saliency maps
- parse GT small objects
- match prediction với GT để xác định TP/FN/localization_error
- tính evidence ratio E_P2/E_P3/E_P4
- tính drop P2→P3 và P3→P4
- lưu CSV và markdown summary

Output bắt buộc:
- artifacts/evidence_analysis/baseline_evidence.csv
- artifacts/evidence_analysis/baseline_evidence_summary.md

Không thêm loss. Không train model mới.
```

---

## Prompt Phase 8 — Evidence loss

```text
Chỉ tiếp tục nếu đã có artifacts/evidence_analysis/baseline_evidence_summary.md.

Hãy triển khai Evidence Preservation Loss.

Tạo:
- src/xai_evidence_sod/losses/evidence_loss.py
- tests/test_evidence_loss.py
- configs/train/evidence_loss_yolo.yaml

Loss v1:
L_evidence = mean(ReLU(E_low - E_high - allowed_drop))

Yêu cầu:
- lambda_evidence = 0 phải tắt effect
- log loss riêng
- test case đơn giản cho loss = 0 và loss > 0
- không phá baseline training script

Chưa cần train full nếu chưa có integration an toàn.
```

---

## Prompt Phase 9 — Train with evidence loss

```text
Hãy tích hợp training với Evidence Preservation Loss.

Tạo:
- scripts/train_with_evidence_loss.py

Yêu cầu:
- đọc configs/train/evidence_loss_yolo.yaml
- train detector với L_total = L_det + lambda_evidence * L_evidence
- log detection loss và evidence loss riêng
- lưu metrics, curves, summary, checkpoint vào experiments/evidence_loss_yolo_seed0/
- nếu có lỗi integration với Ultralytics trainer, tạo custom trainer hoặc callback tối thiểu nhưng không phá baseline

Sau khi hoàn thành, cung cấp command chạy và output kỳ vọng.
```

---

## Prompt Phase 10 — Compare runs

```text
Hãy tạo script so sánh baseline vs evidence loss.

Tạo:
- scripts/compare_runs.py

Script đọc experiments của baseline và evidence loss, sau đó tạo:
- artifacts/comparisons/main_table.csv
- artifacts/comparisons/main_comparison.md
- artifacts/comparisons/evidence_drop_plot.png nếu có dữ liệu

Bảng cần có:
- AP
- AP_S
- AR_S
- FN_small
- E_P2/E_P3/E_P4
- Drop_P2_P3/Drop_P3_P4
- train time nếu có

Không tự bịa metrics nếu file không có. Nếu thiếu dữ liệu, ghi rõ missing.
```
