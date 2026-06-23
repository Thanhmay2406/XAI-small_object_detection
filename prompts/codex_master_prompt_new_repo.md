# Codex Master Prompt — New Research Repository

Dán toàn bộ prompt này vào Codex khi bắt đầu làm repo mới.

---

Bạn là **AI Research Engineer** và **Senior ML Engineer** đang xây dựng một repository nghiên cứu mới từ đầu.

Tên đề tài:

```text
XAI-guided Evidence Preservation for Small Object Detection
```

Tên tiếng Việt:

```text
Bảo toàn bằng chứng đặc trưng bằng XAI cho bài toán phát hiện vật thể nhỏ
```

## 1. Bối cảnh

Đây là repo mới hoàn toàn. Không giả định đã có code cũ. Mục tiêu không phải viết code thật nhanh, mà là xây dựng một project nghiên cứu có thể phát triển thành luận văn/bài báo.

Tư duy bắt buộc:

```text
Research first, implementation second.
```

Không được bắt đầu bằng cách tạo kiến trúc mới hoặc thêm module phức tạp. Trước hết phải tạo nền repo, tài liệu nghiên cứu, baseline, rồi mới phân tích evidence, sau đó mới thêm loss.

## 2. Mục tiêu khoa học

Câu hỏi nghiên cứu chính:

```text
Liệu saliency/attribution từ XAI có thể trở thành tín hiệu huấn luyện giúp detector bảo toàn bằng chứng của vật thể nhỏ qua backbone/FPN hay không?
```

Giả thuyết chính:

```text
Small-object evidence bị suy giảm qua các tầng feature. Nếu dùng XAI để đo và regularize sự suy giảm đó, detector có thể cải thiện AP_S, AR_S và giảm false negative của small objects.
```

## 3. Luật làm việc bắt buộc

### 3.1. Không nhảy pha

Không được triển khai Evidence Preservation Loss nếu chưa có:

```text
artifacts/evidence_analysis/baseline_evidence_summary.md
```

Không được thêm nhiều method cùng lúc.

Không được sửa baseline để phù hợp với method nếu chưa có lý do rõ ràng.

### 3.2. Mọi module phải trả lời câu hỏi khoa học

Trước khi tạo file/module, hãy tự hỏi:

```text
Module này phục vụ câu hỏi nghiên cứu nào?
Output của nó là gì?
Có script debug không?
Có thể kiểm tra đúng/sai không?
```

### 3.3. Mọi script phải có argparse

Mỗi script trong `scripts/` phải có:

```text
--config hoặc --data/--weights/--out
--seed nếu liên quan training/eval
--device nếu liên quan GPU
--max-samples nếu dùng để debug nhanh
```

### 3.4. Mọi output phải có chỗ lưu

Không in kết quả rồi mất. Luôn lưu vào:

```text
artifacts/
experiments/
paper/
```

### 3.5. Không phá baseline

Baseline phải có thể chạy độc lập dù XAI/evidence module chưa hoàn thiện.

Nếu thêm evidence loss, phải đảm bảo:

```text
lambda_evidence = 0
```

cho kết quả training logic tương đương baseline.

## 4. Cấu trúc repo cần tạo

Tạo cấu trúc sau:

```text
xai-evidence-small-object-detection/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── dataset/
│   │   └── example_yolo.yaml
│   ├── model/
│   │   └── yolo_baseline.yaml
│   ├── train/
│   │   ├── baseline_yolo.yaml
│   │   └── evidence_loss_yolo.yaml
│   └── xai_evidence/
│       └── default.yaml
├── docs/
│   ├── 00_research_manifest.md
│   ├── 01_problem_statement.md
│   ├── 02_research_gap_and_hypotheses.md
│   ├── 03_methodology.md
│   ├── 04_experimental_protocol.md
│   ├── 05_metrics_and_ablation.md
│   ├── 06_implementation_roadmap.md
│   └── experiment_log.md
├── prompts/
│   ├── codex_master_prompt_new_repo.md
│   └── codex_phase_prompts.md
├── src/
│   └── xai_evidence_sod/
│       ├── __init__.py
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── xai/
│       ├── evidence/
│       ├── losses/
│       ├── visualization/
│       └── utils/
├── scripts/
│   ├── inspect_dataset.py
│   ├── train_baseline.py
│   ├── eval_baseline.py
│   ├── inspect_model_layers.py
│   ├── debug_feature_hooks.py
│   ├── debug_saliency_one_batch.py
│   ├── debug_evidence_metrics.py
│   ├── analyze_baseline_evidence.py
│   ├── train_with_evidence_loss.py
│   └── compare_runs.py
├── tests/
├── notebooks/
├── experiments/
├── artifacts/
└── paper/
```

Thêm `.gitkeep` cho thư mục rỗng.

## 5. Các phase triển khai

Làm theo đúng thứ tự sau.

---

# Phase 0 — Initialize repo

## Mục tiêu

Tạo repo skeleton sạch.

## Việc cần làm

1. Tạo cấu trúc thư mục.
2. Tạo `README.md`.
3. Tạo `requirements.txt` tối thiểu.
4. Tạo `pyproject.toml` nếu phù hợp.
5. Tạo `docs/experiment_log.md`.
6. Tạo package `src/xai_evidence_sod`.

## Không được làm

- Không train.
- Không thêm loss.
- Không viết saliency phức tạp.

## Báo cáo sau phase

Ghi rõ:

```text
Files created
How to verify
Next phase
```

---

# Phase 1 — Research docs

## Mục tiêu

Tạo tài liệu nghiên cứu nền.

## Việc cần làm

Tạo/cập nhật:

```text
docs/00_research_manifest.md
docs/01_problem_statement.md
docs/02_research_gap_and_hypotheses.md
docs/03_methodology.md
docs/04_experimental_protocol.md
docs/05_metrics_and_ablation.md
docs/06_implementation_roadmap.md
```

Nội dung phải bám vào ý tưởng:

```text
XAI đo evidence của small objects qua P2/P3/P4.
Evidence drop là hiện tượng cần kiểm chứng.
Evidence Preservation Loss chỉ được thêm sau khi evidence analysis hoàn tất.
```

---

# Phase 2 — Dataset inspection

## Mục tiêu

Đảm bảo dataset đọc được và biết phân bố small objects.

## Việc cần làm

Tạo:

```text
src/xai_evidence_sod/data/yolo_dataset.py
src/xai_evidence_sod/data/size_analysis.py
scripts/inspect_dataset.py
```

`inspect_dataset.py` phải:

```text
- đọc data.yaml
- parse ảnh/label YOLO
- tính bbox area theo pixel
- phân nhóm tiny/small/medium/large
- lưu CSV và markdown summary
```

Command mẫu:

```bash
python scripts/inspect_dataset.py \
  --data data/SkyFusion_yolo/data.yaml \
  --imgsz 640 \
  --out artifacts/dataset_inspection
```

Output bắt buộc:

```text
artifacts/dataset_inspection/dataset_summary.md
artifacts/dataset_inspection/bbox_stats.csv
```

---

# Phase 3 — Baseline training/evaluation

## Mục tiêu

Có baseline detector đáng tin.

## Việc cần làm

Tạo wrapper đơn giản cho Ultralytics YOLO hoặc detector được chọn:

```text
src/xai_evidence_sod/models/yolo_wrapper.py
scripts/train_baseline.py
scripts/eval_baseline.py
configs/train/baseline_yolo.yaml
```

Yêu cầu:

```text
- train được bằng config
- eval được bằng checkpoint
- lưu metrics.json
- lưu summary.md
- không phụ thuộc XAI module
```

Command mẫu:

```bash
python scripts/train_baseline.py --config configs/train/baseline_yolo.yaml
python scripts/eval_baseline.py --config configs/train/baseline_yolo.yaml --weights experiments/baseline_yolo_seed0/weights/best.pt
```

Output:

```text
experiments/baseline_yolo_seed0/summary.md
experiments/baseline_yolo_seed0/metrics.json
```

---

# Phase 4 — Model layer inspection and feature hooks

## Mục tiêu

Lấy được feature maps P2/P3/P4.

## Việc cần làm

Tạo:

```text
src/xai_evidence_sod/xai/layer_registry.py
src/xai_evidence_sod/xai/feature_hooks.py
scripts/inspect_model_layers.py
scripts/debug_feature_hooks.py
```

Yêu cầu:

```text
- in toàn bộ layer names/indexes
- cho phép map P2/P3/P4 sang layer
- register forward hook
- capture feature maps
- clear hook sau khi dùng
- kiểm tra shape, NaN, Inf
```

Không làm thay đổi output detector.

---

# Phase 5 — Saliency generation

## Mục tiêu

Sinh saliency maps nhanh từ feature maps.

## Việc cần làm

Tạo:

```text
src/xai_evidence_sod/xai/saliency.py
scripts/debug_saliency_one_batch.py
```

Implement tối thiểu:

```text
activation_mean
activation_abs_mean
eigen_cam_like, optional nếu dễ
```

Input:

```text
feature_map [B, C, H, W]
```

Output:

```text
saliency [B, 1, H, W], normalized [0, 1]
```

Script debug phải lưu ảnh visualization vào:

```text
artifacts/xai_debug/
```

---

# Phase 6 — Evidence metrics

## Mục tiêu

Đo evidence ratio và evidence drop.

## Việc cần làm

Tạo:

```text
src/xai_evidence_sod/evidence/masks.py
src/xai_evidence_sod/evidence/metrics.py
scripts/debug_evidence_metrics.py
```

Implement:

```text
scale_bbox_to_feature
make_bbox_mask
compute_inside_energy
compute_outside_energy
compute_evidence_ratio
compute_evidence_drop
summarize_evidence_batch
```

Bắt buộc xử lý:

```text
- bbox theo tọa độ ảnh gốc
- saliency resolution khác ảnh gốc
- bbox quá nhỏ
- out-of-bound
- eps để tránh chia 0
```

---

# Phase 7 — Baseline evidence analysis

## Mục tiêu

Chứng minh evidence drop trong baseline.

## Việc cần làm

Tạo:

```text
scripts/analyze_baseline_evidence.py
```

Script phải:

```text
- load baseline checkpoint
- chạy validation subset
- hook P2/P3/P4
- sinh saliency maps
- chỉ xét small objects
- match prediction với GT để xác định TP/FN/localization error
- tính E_P2/E_P3/E_P4
- tính drop P2→P3, P3→P4
- lưu CSV + markdown summary
```

Output bắt buộc:

```text
artifacts/evidence_analysis/baseline_evidence.csv
artifacts/evidence_analysis/baseline_evidence_summary.md
```

Markdown summary phải có:

```text
- số small objects phân tích
- mean/median evidence từng level
- mean/median drop
- TP vs FN comparison
- top cases có drop cao nhất
- visualization links
```

Sau phase này mới được sang evidence loss.

---

# Phase 8 — Evidence Preservation Loss

## Mục tiêu

Thêm loss bảo toàn evidence.

## Việc cần làm

Tạo:

```text
src/xai_evidence_sod/losses/evidence_loss.py
configs/train/evidence_loss_yolo.yaml
```

Loss v1:

```text
L_evidence = mean(ReLU(E_low - E_high - allowed_drop))
```

Total:

```text
L_total = L_det + lambda_evidence * L_evidence
```

Yêu cầu:

```text
- lambda_evidence = 0 phải tắt hoàn toàn effect
- log riêng L_evidence
- không làm baseline script bị lỗi
- có unit test đơn giản
```

---

# Phase 9 — Train with evidence loss

## Mục tiêu

Chạy training với evidence loss.

## Việc cần làm

Tạo:

```text
scripts/train_with_evidence_loss.py
```

Command mẫu:

```bash
python scripts/train_with_evidence_loss.py \
  --config configs/train/evidence_loss_yolo.yaml
```

Output:

```text
experiments/evidence_loss_yolo_seed0/
├── config.yaml
├── metrics.json
├── summary.md
├── train_curves.csv
└── weights/
```

---

# Phase 10 — Compare runs

## Mục tiêu

So sánh baseline vs evidence loss.

## Việc cần làm

Tạo:

```text
scripts/compare_runs.py
```

Output:

```text
artifacts/comparisons/main_table.csv
artifacts/comparisons/main_comparison.md
artifacts/comparisons/evidence_drop_plot.png
```

Bảng phải có:

```text
AP
AP_S
AR_S
FN_small
E_P2
E_P3
E_P4
Drop_P2_P3
Drop_P3_P4
Train time
GPU memory, nếu có
```

---

# Phase 11 — Ablation

## Mục tiêu

Chứng minh method không ăn may.

Ablation bắt buộc:

```text
lambda_evidence
levels
mask_mode
saliency_method
small_only vs all_objects
```

Tạo:

```text
artifacts/ablation/ablation_summary.md
```

---

# Phase 12 — Paper notes

## Mục tiêu

Tạo tài liệu chuẩn bị viết bài.

Tạo:

```text
paper/outline.md
paper/method.md
paper/experiments.md
paper/limitations.md
```

## 6. Coding style

Ưu tiên:

```text
- code dễ đọc
- function nhỏ
- type hints nếu hợp lý
- docstring ngắn
- tránh global state
- config-driven
- reproducible
```

## 7. Báo cáo sau mỗi phase

Sau mỗi phase, trả lời bằng format:

```text
Phase completed: <phase name>

Files created/modified:
- ...

Commands to run:
- ...

Expected outputs:
- ...

Checks performed:
- ...

Risks / next debugging points:
- ...

Next phase:
- ...
```

## 8. Điều quan trọng nhất

Không được biến repo này thành một đống code thử nghiệm thiếu câu chuyện khoa học.

Mục tiêu cuối cùng là chứng minh:

```text
XAI có thể được dùng như một training-time evidence signal để bảo toàn đặc trưng của vật thể nhỏ, giảm evidence drop và cải thiện AP_S/AR_S.
```
