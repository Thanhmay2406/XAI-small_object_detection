# XAI-guided Evidence Preservation for Small Object Detection

> Repository nghiên cứu mới cho đề tài: **Bảo toàn bằng chứng đặc trưng bằng XAI cho bài toán phát hiện vật thể nhỏ**.

## 1. Mục tiêu ngắn gọn

Đề tài này nghiên cứu việc dùng Explainable AI (XAI) không chỉ để hậu kiểm detector, mà như một **training signal** để đo, giám sát và bảo toàn “bằng chứng” của vật thể nhỏ trong quá trình trích xuất đặc trưng qua backbone/FPN.

Câu hỏi chính:

> Liệu saliency/attribution có thể trở thành tín hiệu huấn luyện giúp detector bảo toàn bằng chứng của vật thể nhỏ qua các tầng feature và cải thiện AP_S/AR_S hay không?

## 2. Tư duy nghiên cứu

Repo này không bắt đầu bằng việc tạo kiến trúc mới. Quy trình đúng là:

```text
Observation
  ↓
Scientific Question
  ↓
Hypothesis
  ↓
Diagnostic Evidence
  ↓
Method
  ↓
Experiment
  ↓
Paper-ready Evidence
```

Sai lầm cần tránh:

```text
Có ý tưởng → tạo module mới → train ngay → kết quả khó giải thích
```

Cách làm mong muốn:

```text
Có giả thuyết → đo hiện tượng → chứng minh có evidence drop → mới thiết kế loss
```

## 3. Bốn stage chính

```text
Stage 1 — Research Foundation
Stage 2 — Baseline Construction
Stage 3 — Evidence Analysis
Stage 4 — Evidence-guided Learning
```

### Stage 1 — Research Foundation

Tạo nền tảng khoa học của đề tài:

- Problem statement
- Research gap
- Research questions
- Hypotheses
- Experimental protocol
- Metric definitions
- Repo structure

### Stage 2 — Baseline Construction

Xây baseline object detector sạch, có thể lặp lại:

- Dataset loader
- YOLO/RT-DETR/Faster R-CNN wrapper hoặc Ultralytics wrapper
- Training config
- Evaluation script
- Logging
- Baseline report

Không thêm XAI/loss ở stage này.

### Stage 3 — Evidence Analysis

Dùng XAI để đo hiện tượng:

- Hook feature maps P2/P3/P4
- Sinh saliency maps
- Đo evidence trong bbox small object
- Đo evidence outside bbox
- Đo evidence drop qua tầng
- Kiểm tra correlation giữa evidence drop và lỗi detector

Không train model mới ở stage này.

### Stage 4 — Evidence-guided Learning

Sau khi chứng minh hiện tượng evidence drop, mới thêm:

- Evidence Preservation Loss
- Saliency Alignment Loss, nếu cần
- Cross-layer Evidence Consistency
- Ablation study
- So sánh với baseline và augmentation truyền thống

## 4. Cấu trúc repo đề xuất

```text
xai-evidence-small-object-detection/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── dataset/
│   ├── model/
│   ├── train/
│   └── xai_evidence/
├── docs/
│   ├── 00_research_manifest.md
│   ├── 01_problem_statement.md
│   ├── 02_research_gap_and_hypotheses.md
│   ├── 03_methodology.md
│   ├── 04_experimental_protocol.md
│   ├── 05_metrics_and_ablation.md
│   └── 06_implementation_roadmap.md
├── prompts/
│   ├── codex_master_prompt_new_repo.md
│   └── codex_phase_prompts.md
├── src/
│   └── xai_evidence_sod/
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
│   ├── debug_feature_hooks.py
│   ├── analyze_baseline_evidence.py
│   ├── train_with_evidence_loss.py
│   └── compare_runs.py
├── experiments/
├── artifacts/
├── notebooks/
├── tests/
└── paper/
```

## 5. Kết quả kỳ vọng

Kết quả không chỉ là heatmap đẹp. Cần chứng minh bằng metric:

```text
AP_S ↑
AR_S ↑
Recall small object ↑
False Negative small object ↓
Evidence ratio inside bbox ↑
Evidence outside bbox ↓
Evidence drop P2→P3/P3→P4 ↓
Training stability ↑
```

## 6. Quy tắc làm việc

1. Không thêm loss nếu chưa có báo cáo evidence analysis.
2. Không sửa baseline khi đang thêm module nghiên cứu.
3. Mọi thí nghiệm phải có config riêng.
4. Mọi script phải có output rõ ràng.
5. Mọi kết quả phải lưu vào `experiments/` hoặc `artifacts/`.
6. Mọi kết luận khoa học phải có bảng số liệu hoặc visualization đi kèm.
7. Ưu tiên code nhỏ, kiểm thử được, hơn là code lớn nhưng khó debug.

## 7. File quan trọng nên đọc trước

```text
docs/00_research_manifest.md
docs/06_implementation_roadmap.md
prompts/codex_master_prompt_new_repo.md
prompts/codex_phase_prompts.md
```
