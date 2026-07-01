# Đặc tả phương pháp M8_v1c / M8_v1c.2

**Tên phương pháp đề xuất:** Khai thác evidence bằng XAI/activation để tự sinh policy, kết hợp instance-aware spatial scale weighting cho YOLO.

**File triển khai khuyến nghị:** `scripts/run_scale_pipeline.py`

**Đường dẫn tài liệu khuyến nghị trong repo:** `docs/m8_note.md`

---

## 1. Mục tiêu và động lực

Ý tưởng scale-weighting ban đầu của M8 là cải thiện YOLO bằng cách gán từng nhóm kích thước object cho một hoặc vài feature scale phù hợp hơn, ví dụ `P2`, `P3`, `P4`, `P5`.

Ở các phiên bản trước, policy thường được suy ra một phần từ quan sát thủ công trên XAI hoặc activation evidence, ví dụ:

```text
small  -> P3
medium -> P5
large  -> P5
```

Cách này hữu ích ở giai đoạn khám phá, nhưng có một số điểm yếu:

1. Policy phụ thuộc vào cách con người diễn giải evidence.
2. Khó tái lập chính xác quyết định chọn scale.
3. Khó audit bằng số liệu định lượng.
4. Một policy ở mức toàn ảnh là quá thô nếu ảnh chứa nhiều object có kích thước khác nhau.
5. Các hệ số trong công thức evidence score, ví dụ `0.5 / 0.3 / 0.2`, ban đầu vẫn là heuristic.

Phương pháp mới được thiết kế để chuyển workflow thành một pipeline có kiểm soát, có artifact, có guardrail và có khả năng fail-closed:

```text
baseline YOLO
    -> khai thác multi-scale activation / XAI evidence
    -> tính object-scale evidence metrics
    -> kiểm tra độ nhạy / hiệu chỉnh hệ số
    -> sinh size-aware policy candidate
    -> guardrail + identity fallback
    -> instance-aware spatial runtime weighting
    -> dry-run preflight
    -> tùy chọn training có kiểm soát
```

Ý tưởng lõi không chỉ là thêm một module scale-weighting. Cách hiểu đúng hơn là:

```text
policy mining first, runtime weighting second
```

Tức là trước hết dùng XAI/activation để khai thác evidence và tự sinh policy, sau đó mới đưa policy vào runtime training.

---

## 2. Tóm tắt phương pháp ở mức cao

Phương pháp dùng một baseline YOLO đã train làm nguồn evidence. Pipeline hook activation ở các feature scale `P2`, `P3`, `P4`, `P5`, đo mức độ mỗi scale tập trung vào từng ground-truth object, tổng hợp evidence theo nhóm kích thước object, rồi sinh ra một runtime policy bảo thủ.

Sau đó, policy được áp dụng trong training bằng một cơ chế instance-aware spatial weighting. Thay vì áp một policy cho toàn bộ ảnh, mỗi GT object được project xuống từng feature map, tạo spatial mask, và chỉ vùng không gian tương ứng với object đó mới được áp policy của size group tương ứng.

Tóm tắt ngắn:

```text
M8_v1c.2 = XAI/activation evidence mining
        -> calibrated size-aware policy
        -> instance-aware spatial feature weighting
        -> controlled YOLO training
```

---

## 3. Claim nghiên cứu an toàn

Claim an toàn nên dùng:

> Chúng tôi đề xuất một pipeline XAI-guided evidence-to-policy tự động cho size-aware multi-scale feature weighting trong YOLO. Phương pháp khai thác object-scale activation evidence từ baseline detector đã train, suy ra scale policy bảo thủ theo nhóm kích thước object, fallback về identity khi evidence không đủ chắc chắn, và đưa policy vào training thông qua instance-aware spatial runtime weighting để kiểm chứng downstream một cách có kiểm soát.

Không nên claim rằng phương pháp tìm ra scale tối ưu tuyệt đối cho từng kích thước object.

Cách diễn giải an toàn hơn:

```text
Phương pháp xác định scale có evidence mạnh nhất theo metric đã định nghĩa,
rồi kiểm chứng downstream xem mined policy đó có cải thiện detection behavior hay không.
```

---

## 4. Các vấn đề chính được giải quyết

### 4.1. Chọn policy thủ công

Logic cũ phụ thuộc vào việc người nghiên cứu nhìn evidence rồi quyết định scale nào nên được ưu tiên cho từng size group.

Pipeline mới thay thế bước này bằng evidence mining có cấu trúc:

```text
object x scale metrics
    -> aggregation theo size_group x scale
    -> policy candidate generation
```

Nhờ đó, policy có thể được audit bằng CSV/JSON/YAML thay vì chỉ dựa vào quan sát.

### 4.2. Hệ số evidence score còn heuristic

Công thức evidence score ban đầu là:

```text
evidence_score =
    0.5 * energy_in_box
  + 0.3 * peak_alignment_score
  + 0.2 * (1 - background_leakage)
```

Các hệ số này nên được xem là default heuristic, không phải chân lý cuối cùng.

Phương pháp nên có ít nhất một trong hai cơ chế:

1. sensitivity analysis cho nhiều bộ hệ số khác nhau;
2. coefficient calibration trên một calibration split riêng.

### 4.3. Image-level policy quá thô

Nếu một ảnh chứa cả object nhỏ và object lớn, việc áp một policy duy nhất cho toàn bộ feature map có thể gây bias sai.

Ví dụ:

```text
ảnh có:
  object nhỏ bên trái
  object lớn bên phải

image-level policy chọn small
  -> P3 được boost trên toàn ảnh
  -> vùng object lớn cũng bị nhận small-policy
```

Phương pháp mới khắc phục bằng instance-aware spatial masks:

```text
object -> size_group -> projected feature-map mask -> local scale weighting
```

### 4.4. Can thiệp không an toàn khi evidence yếu

Nếu evidence yếu hoặc mơ hồ, phương pháp không được ép chọn scale.

Giải pháp là:

```text
identity fallback
```

Identity nghĩa là:

```text
P2 = 1.0
P3 = 1.0
P4 = 1.0
P5 = 1.0
```

Tức là giữ nguyên hành vi baseline, không can thiệp.

---

## 5. Input của all-in-one pipeline

All-in-one script nên nhận các input sau:

```text
--dataset-yaml      đường dẫn tới YOLO dataset YAML
--checkpoint        đường dẫn baseline checkpoint
--model-yaml        tùy chọn: đường dẫn YOLO model YAML
--output-dir        thư mục ghi artifact
--imgsz             kích thước input image
--max-images        tùy chọn: giới hạn số ảnh để analysis
--batch-size        tùy chọn: batch size
--stage             analysis | calibrate | export | dry-run | train
--execute           chỉ bắt buộc khi train thật
```

Ví dụ dry-run:

```bash
python scripts/run_scale_pipeline.py \
  --dataset-yaml configs/dataset/drill_bit_yolo.yaml \
  --checkpoint experiments/yolov8_p2/yolov8_p2_run/weights/best.pt \
  --output-dir artifacts/scale_pipeline \
  --stage dry-run
```

Training thật bắt buộc phải có:

```bash
--stage train --execute
```

---

## 6. Output artifacts

Pipeline nên ghi toàn bộ output vào một artifact directory riêng, ví dụ:

```text
artifacts/scale_pipeline/
```

Các output khuyến nghị:

```text
object_scale_scores.csv
scale_evidence_metrics.csv
coefficient_sensitivity.csv
coefficient_calibration_summary.json
policy_candidate.yaml
runtime_policy.yaml
spatial_runtime_policy_preview.json
runtime_validation_summary.json
non_execution_manifest.json
execution_manifest.json
method_summary.json
method_report.md
README.md
```

Artifact directory phải ghi rõ training có được execute hay không:

```text
training_executed: true / false
```

---

## 7. Stage A — Dùng baseline detector làm nguồn evidence

Phương pháp bắt đầu từ một baseline YOLO checkpoint đã train.

Baseline này chỉ được dùng để trích xuất activation evidence trong analysis stage.

Ở stage này, pipeline không được:

```text
train
mutate dataset
mutate checkpoint
overwrite baseline weights
run uncontrolled prediction/export
```

Baseline được xem như một evidence generator:

```text
image -> YOLO baseline -> P2/P3/P4/P5 activations
```

---

## 8. Stage B — Hook multi-scale activation

Script cần xác định các feature level được Detect head sử dụng.

Ví dụ mapping:

```text
P2 -> layer 18
P3 -> layer 21
P4 -> layer 24
P5 -> layer 27
```

Layer index chính xác nên được resolve từ model structure hoặc truyền qua config.

Forward hook được đăng ký để khi model forward qua các layer này, script lưu lại:

```text
activation[P2]
activation[P3]
activation[P4]
activation[P5]
```

Mỗi activation tensor có shape tương tự:

```text
[B, C, H_l, W_l]
```

Trong đó `l` là feature level.

---

## 9. Stage C — Đọc GT object và chia size group

Script đọc YOLO label và chuyển normalized coordinates sang pixel-space box:

```text
class_id x_center y_center width height
    ->
x1 y1 x2 y2
```

Mỗi object được gán vào một size group dựa trên area của bounding box.

Threshold mặc định đề xuất:

```text
tiny   : area < 32 * 32
small  : area < 96 * 96
medium : area < 224 * 224
large  : còn lại
```

Nhóm `tiny` có thể được giữ riêng trong analysis. Ở runtime implementation đầu tiên, có thể merge `tiny` vào `small` hoặc fallback `tiny` về identity nếu evidence chưa đủ.

---

## 10. Stage D — Tính object-scale evidence metrics

Với mỗi object và mỗi scale, phương pháp tính các evidence metrics.

Đơn vị phân tích lõi là:

```text
object_i x scale_l
```

Ví dụ:

```text
object_001 x P2
object_001 x P3
object_001 x P4
object_001 x P5
```

### 10.1. energy_in_box

Metric này đo lượng activation energy nằm bên trong GT box.

Ý tưởng:

```text
energy_in_box = activation_inside_gt_box / total_activation
```

Giá trị cao nghĩa là scale đó đang tập trung tốt vào vùng object.

### 10.2. background_leakage

Metric này đo lượng activation bị loang ra ngoài vùng object.

Ý tưởng:

```text
background_leakage = activation_outside_gt_box / total_activation
```

Giá trị thấp là tốt.

Trong evidence score, ta dùng:

```text
1 - background_leakage
```

### 10.3. peak_alignment_score

Metric này đo xem điểm activation mạnh nhất có gần object hay không.

Giá trị cao nghĩa là hotspot của activation nằm gần tâm object hoặc bên trong object region.

### 10.4. evidence_score

Default heuristic:

```text
evidence_score =
    0.5 * energy_in_box
  + 0.3 * peak_alignment_score
  + 0.2 * (1 - background_leakage)
```

Score này dùng để xếp hạng scale cho từng object.

Các hệ số phải cấu hình được:

```yaml
metric_weights:
  energy_in_box: 0.5
  peak_alignment_score: 0.3
  anti_background_leakage: 0.2
```

---

## 11. Stage E — Sensitivity analysis và calibration cho hệ số

Các hệ số `0.5 / 0.3 / 0.2` không nên được xem là kết luận cuối cùng.

Pipeline nên hỗ trợ hai mode.

### 11.1. Sensitivity analysis

Chạy nhiều bộ hệ số thỏa mãn:

```text
w_energy >= 0
w_peak >= 0
w_anti_leakage >= 0
w_energy + w_peak + w_anti_leakage = 1
```

Với mỗi bộ hệ số:

```text
1. tính lại evidence_score
2. aggregate theo size_group x scale
3. mine policy candidate
4. so sánh độ ổn định của policy
```

Nếu nhiều bộ hệ số khác nhau vẫn cho cùng policy, policy đó đáng tin hơn.

### 11.2. Calibration split

Phiên bản mạnh hơn là dùng calibration split.

Pipeline tìm bộ hệ số tạo ra policy ổn định và hữu ích nhất trên calibration data, sau đó đánh giá policy đã chọn trên validation/test data.

Quy tắc khuyến nghị:

```text
calibration split -> chọn hệ số
validation/test split -> báo kết quả cuối
```

Không nên vừa chọn hệ số vừa báo final performance trên cùng một tập dữ liệu.

---

## 12. Stage F — Aggregate theo size_group x scale

Sau khi có object-scale scores, aggregate theo:

```text
size_group x scale
```

Với mỗi bucket, tính:

```text
object_count
mean_energy_in_box
mean_background_leakage
mean_peak_alignment_score
mean_evidence_score
std_evidence_score
```

Ví dụ:

```text
small  x P2 -> mean_evidence_score = 0.41
small  x P3 -> mean_evidence_score = 0.58
small  x P4 -> mean_evidence_score = 0.36
small  x P5 -> mean_evidence_score = 0.22
```

Bảng này hỗ trợ candidate policy:

```text
small -> P3
```

---

## 13. Stage G — Sinh policy candidate

Rule chọn cơ bản là:

```text
policy(size_group) = argmax_scale mean_evidence_score(size_group, scale)
```

Nhưng chỉ argmax là chưa đủ. Pipeline bắt buộc phải có guardrail.

Guardrail khuyến nghị:

```text
min_object_count
min_best_second_margin
min_bootstrap_stability
allowed_scale_set
valid_metric_range
```

Với mỗi size group:

```text
if object_count < min_object_count:
    policy = identity

elif best_score - second_best_score < min_margin:
    policy = identity

elif bootstrap_stability < min_stability:
    policy = identity

else:
    policy = selected best scale
```

Ví dụ policy candidate:

```yaml
small:
  selected_scale: P3
  status: accepted
  reason: sufficient_object_count_and_margin

medium:
  selected_scale: identity
  status: fallback
  reason: insufficient_margin

large:
  selected_scale: P5
  status: accepted
  reason: sufficient_object_count_and_margin

unknown:
  selected_scale: identity
  status: fixed_identity
  reason: unknown_or_mixed_case
```

---

## 14. Stage H — Export runtime policy

Policy candidate là output cho analysis. Runtime training cần schema đơn giản hơn.

Ví dụ runtime policy:

```yaml
method: m8_v1c_instance_aware_spatial_scale_weighting
feature_levels: [P2, P3, P4, P5]

scale_weights:
  small:
    P2: 1.0
    P3: 1.2
    P4: 1.0
    P5: 1.0

  medium:
    P2: 1.0
    P3: 1.0
    P4: 1.0
    P5: 1.0

  large:
    P2: 1.0
    P3: 1.0
    P4: 1.0
    P5: 1.2

  unknown:
    P2: 1.0
    P3: 1.0
    P4: 1.0
    P5: 1.0
```

Bất kỳ size group nào không đủ chắc chắn đều phải export thành identity.

Nhóm `unknown` luôn luôn là identity.

---

## 15. Vì sao image-level policy là chưa đủ

Feature map của YOLO là image-level:

```text
F_l in R^{B x C x H_l x W_l}
```

Nhưng GT box là instance-level:

```text
object_1 -> small
object_2 -> large
object_3 -> medium
```

Nếu một ảnh chứa cả object nhỏ và lớn, việc chọn một policy duy nhất cho toàn ảnh là quá thô.

Ví dụ vấn đề:

```text
image contains:
  small object on the left
  large object on the right

image-level policy chooses small
  -> P3 boosted everywhere
  -> large-object region also receives small-object policy
```

Điều này có thể đưa bias sai vào training.

---

## 16. Stage I — Instance-aware spatial scale weighting

Giải pháp đề xuất là instance-aware spatial weighting.

Thay vì áp một policy cho toàn feature map, phương pháp chỉ áp policy bên trong vùng object.

Với mỗi GT object:

```text
1. xác định size_group
2. project GT box xuống từng feature map scale
3. tạo spatial mask
4. áp policy của size_group đó chỉ bên trong mask
```

### 16.1. Ký hiệu feature

Với feature level `l`:

```text
F_l in R^{B x C x H_l x W_l}
```

Với image `b`, size group `g`, và feature level `l`, tạo mask:

```text
M_{b,g,l} in R^{1 x H_l x W_l}
```

### 16.2. Project box xuống feature map

Cho object box ở input-image pixel coordinates:

```text
x1, y1, x2, y2
```

Project xuống feature level `l` bằng stride:

```text
x1_l = x1 / stride_l
y1_l = y1 / stride_l
x2_l = x2 / stride_l
y2_l = y2 / stride_l
```

Sau đó fill vùng tương ứng trong mask.

### 16.3. Công thức spatial weighting

Gọi runtime policy weight cho size group `g` và scale `l` là:

```text
w_{g,l}
```

Định nghĩa:

```text
beta_{g,l} = w_{g,l} - 1
```

Xây dựng spatial adjustment map:

```text
A_l[b] = 1 + sum_g beta_{g,l} * M_{b,g,l}
```

Áp vào feature:

```text
F'_l[b] = F_l[b] * A_l[b]
```

Ý nghĩa:

```text
small-object region  -> nhận small policy
large-object region  -> nhận large policy
background           -> identity
unknown/mixed region -> identity hoặc safe clamped combination
```

---

## 17. Các lựa chọn mask construction

### 17.1. Binary box mask

Phiên bản đơn giản nhất:

```text
mask = 1 bên trong projected GT box
mask = 0 bên ngoài
```

Khuyến nghị dùng cho implementation đầu tiên.

### 17.2. Soft Gaussian mask

Phiên bản mềm hơn:

```text
mask mạnh nhất gần tâm object
mask yếu hơn gần biên object
mask bằng 0 ở background hoặc vùng xa object
```

Cách này có thể giảm quantization artifact cho object nhỏ.

Chỉ nên thử sau khi binary-mask version chạy ổn.

### 17.3. Xử lý overlap

Nếu nhiều object overlap trên cùng feature cell, phiên bản đầu tiên nên dùng:

```text
mask = clamp(sum(object_masks), 0, 1)
```

Hoặc:

```text
mask = max(object_masks)
```

Không nên cho boost cộng dồn không giới hạn.

---

## 18. Training-time và inference-time behavior

Một vấn đề quan trọng:

```text
training có GT boxes
inference không có GT boxes
```

Nếu spatial mask được tạo từ GT box, weighting này tự nhiên là training-time guidance.

Implementation đầu tiên khuyến nghị:

```text
training: dùng GT-based spatial policy weighting
validation/inference: identity runtime path, trừ khi đã có learned gate
```

Cách này tránh yêu cầu GT box ở inference.

Tuy nhiên, nó có thể tạo train-test mismatch. Vì vậy evaluation phải có các control:

```text
baseline YOLO
identity runtime control
M8_v1c image-level policy
M8_v1c.2 instance-aware spatial policy
```

Nếu train-test mismatch gây hại, nâng cấp tiếp theo nên là learned spatial gate.

---

## 19. Nâng cấp tương lai — learned spatial gate

Phiên bản mạnh hơn có thể học một gate:

```text
F_l -> Gate_l
```

Trong đó:

```text
Gate_l in R^{B x 1 x H_l x W_l}
```

Sau đó:

```text
F'_l = F_l * (1 + alpha * Gate_l)
```

Gate có thể được supervise bằng GT-based masks hoặc XAI-mined masks trong training, nhưng ở inference gate được dự đoán trực tiếp từ feature.

Điều này giảm train-test mismatch, nhưng tăng độ phức tạp triển khai.

---

## 20. Thiết kế all-in-one implementation

Pipeline có thể triển khai trong một file Python, nhưng file đó nên đóng vai trò orchestrator với các function rõ ràng.

File khuyến nghị:

```text
scripts/run_scale_pipeline.py
```

Function layout khuyến nghị:

```python
def parse_args():
    ...

def load_dataset_config():
    ...

def load_baseline_model():
    ...

def resolve_feature_scales():
    ...

def register_activation_hooks():
    ...

def collect_object_scale_metrics():
    ...

def compute_evidence_score():
    ...

def run_coefficient_sensitivity():
    ...

def aggregate_metrics_by_size_group():
    ...

def mine_policy_candidate():
    ...

def export_runtime_policy():
    ...

def validate_runtime_policy():
    ...

def build_instance_spatial_masks():
    ...

def apply_instance_aware_scale_weighting():
    ...

def dry_run_runtime_integration():
    ...

def train_if_execute():
    ...

def write_artifacts():
    ...

def main():
    ...
```

File không nên là một monolithic script lộn xộn. Nó nên là single entrypoint nhưng bên trong có stage rõ ràng.

---

## 21. Các execution stages

All-in-one script nên hỗ trợ staged execution.

### 21.1. analysis

```bash
python scripts/run_scale_pipeline.py \
  --stage analysis \
  --dataset-yaml configs/dataset/drill_bit_yolo.yaml \
  --checkpoint experiments/yolov8_p2/yolov8_p2_run/weights/best.pt \
  --output-dir artifacts/scale_pipeline
```

Chạy:

```text
activation hook
object-scale metric mining
aggregation
policy candidate generation
```

Không train.

### 21.2. calibrate

```bash
python scripts/run_scale_pipeline.py \
  --stage calibrate \
  --dataset-yaml configs/dataset/drill_bit_yolo.yaml \
  --checkpoint experiments/yolov8_p2/yolov8_p2_run/weights/best.pt \
  --output-dir artifacts/scale_pipeline
```

Chạy coefficient sensitivity hoặc calibration.

Không train.

### 21.3. export

```bash
python scripts/run_scale_pipeline.py \
  --stage export \
  --dataset-yaml configs/dataset/drill_bit_yolo.yaml \
  --checkpoint experiments/yolov8_p2/yolov8_p2_run/weights/best.pt \
  --output-dir artifacts/scale_pipeline
```

Export runtime policy YAML.

Không train.

### 21.4. dry-run

```bash
python scripts/run_scale_pipeline.py \
  --stage dry-run \
  --dataset-yaml configs/dataset/drill_bit_yolo.yaml \
  --checkpoint experiments/yolov8_p2/yolov8_p2_run/weights/best.pt \
  --output-dir artifacts/scale_pipeline
```

Kiểm tra:

```text
runtime policy schema
feature-level mapping
spatial mask construction
safe forward pass compatibility
artifact writing
non-execution manifest
```

Không train.

### 21.5. train

```bash
python scripts/run_scale_pipeline.py \
  --stage train \
  --execute \
  --dataset-yaml configs/dataset/drill_bit_yolo.yaml \
  --checkpoint experiments/yolov8_p2/yolov8_p2_run/weights/best.pt \
  --output-dir artifacts/scale_pipeline
```

Training thật chỉ được chạy khi cả hai điều kiện đúng:

```text
stage == train
--execute có mặt
```

Nếu không, script phải fail-closed.

---

## 22. Safety và guardrails

Script phải mặc định ở trạng thái non-execution.

Guardrail bắt buộc:

```text
1. Không train nếu thiếu --stage train --execute.
2. Không overwrite checkpoint.
3. Không mutate dataset.
4. Unknown policy luôn identity.
5. Evidence không đủ chắc chắn thì fallback identity.
6. Mask mixed hoặc invalid phải được clamp an toàn.
7. Runtime policy phải pass schema validation.
8. Tất cả output ghi vào artifact directory riêng.
9. Non-execution manifest luôn được ghi cho non-training stages.
10. Execution manifest chỉ được ghi khi training thật sự chạy.
```

Fail-closed statuses khuyến nghị:

```text
m8_v1c_blocked_missing_execute_flag
m8_v1c_blocked_invalid_dataset_yaml
m8_v1c_blocked_missing_checkpoint
m8_v1c_blocked_invalid_runtime_policy
m8_v1c_blocked_unresolved_feature_scales
m8_v1c_analysis_completed_no_training
m8_v1c_dry_run_passed_no_training
m8_v1c_training_executed
```

---

## 23. Kết quả kỳ vọng

### 23.1. Kết quả ở tầng analysis

Kỳ vọng:

```text
object-scale evidence table rõ ràng
size_group x scale aggregate table
policy candidate có guardrail decision
```

### 23.2. Kết quả ở tầng policy

Best case:

```text
small  -> P3
medium -> identity hoặc P4/P5
large  -> P5
unknown -> identity
```

Acceptable case:

```text
một số group accepted
một số group identity
```

Still valid case:

```text
tất cả group identity
```

Điều này có nghĩa là evidence chưa đủ mạnh để can thiệp an toàn.

### 23.3. Kết quả ở tầng training

Kỳ vọng thực tế:

```text
overall mAP không giảm
AP-small cải thiện nếu small policy được accepted
convergence speed có thể tốt hơn
random/shuffled/inverted policy kém hơn mined policy
```

Không nên kỳ vọng global mAP tăng rất lớn.

---

## 24. Evaluation plan

Phương pháp nên được đánh giá với các control sau:

```text
Baseline YOLO
Identity runtime policy
M8_v1b manual policy
M8_v1c image-level mined policy
M8_v1c.2 instance-aware spatial policy
Random policy
Shuffled policy
Inverted policy
```

Metric:

```text
mAP50
mAP50-95
AP-small
AP-medium
AP-large
class-wise AP
false negative rate
convergence speed
training stability
multi-seed mean/std
```

Evidence quan trọng nhất không chỉ là mAP tổng cao hơn. Kết quả mạnh hơn là:

```text
Improvement nên xuất hiện đúng ở size group mà mined policy target.
```

Ví dụ:

```text
nếu small -> P3 được accepted,
thì AP-small nên cải thiện rõ hơn AP-large.
```

---

## 25. Ablation plan

### 25.1. Policy source ablation

```text
manual policy vs mined policy vs random policy vs identity
```

### 25.2. Runtime type ablation

```text
image-level weighting vs instance-aware spatial weighting
```

### 25.3. Coefficient ablation

```text
0.5 / 0.3 / 0.2
0.4 / 0.4 / 0.2
0.6 / 0.2 / 0.2
1.0 / 0.0 / 0.0
0.0 / 1.0 / 0.0
0.0 / 0.0 / 1.0
```

### 25.4. Boost strength ablation

```text
1.05
1.10
1.20
1.30
```

### 25.5. Mask type ablation

```text
binary box mask
soft Gaussian mask
identity mask control
```

---

## 26. Rủi ro và cách giảm thiểu

### Risk 1 — evidence score không phải causal proof

Activation evidence không chứng minh chắc chắn rằng boost một scale sẽ cải thiện detection.

Mitigation:

```text
validate bằng downstream training
so sánh với random/shuffled/inverted policies
```

### Risk 2 — overfit hệ số

Bộ hệ số được chọn có thể overfit calibration subset.

Mitigation:

```text
dùng calibration split tách khỏi validation/test
báo sensitivity analysis
```

### Risk 3 — train-test mismatch

GT-based spatial weighting có trong training nhưng không có ở inference.

Mitigation:

```text
so sánh với identity inference
monitor validation performance
cân nhắc learned spatial gate như future work
```

### Risk 4 — boost quá mạnh làm training không ổn định

Mitigation:

```text
bắt đầu với boost nhỏ, ví dụ 1.05 hoặc 1.10
clamp spatial adjustment map
fallback identity khi evidence không chắc
```

### Risk 5 — ảnh mixed-size tạo mask xung đột

Mitigation:

```text
instance-aware spatial masks
clamp overlapping regions
unknown identity fallback
```

---

## 27. Scope triển khai đầu tiên khuyến nghị

Implementation đầu tiên nên giữ phạm vi bảo thủ:

```text
1. Một all-in-one script.
2. Hỗ trợ các stage: analysis, calibrate, export, dry-run, train.
3. Chỉ dùng binary spatial masks.
4. GT-based instance-aware weighting chỉ trong training.
5. Unknown và uncertain policies fallback identity.
6. Chưa dùng learned gate.
7. Không mutate dataset.
8. Không overwrite checkpoint.
```

Scope này vừa đủ để giải quyết điểm yếu lớn nhất của image-level policy mà vẫn kiểm soát được rủi ro.

---

## 28. Tên phương pháp đề xuất

Tên khuyến nghị:

```text
M8_v1c.2 Instance-Aware XAI-Guided Spatial Scale Policy
```

Tên ngắn hơn:

```text
M8_v1d Instance-Aware Spatial Scale Weighting
```

Nếu muốn giữ lineage với M8_v1c, dùng:

```text
M8_v1c.2
```

Nếu muốn tách thành một branch phương pháp mới sạch hơn, dùng:

```text
M8_v1d
```

---

## 29. Mô tả một đoạn hoàn chỉnh

M8_v1c.2 là một pipeline XAI-guided evidence-to-policy tự động cho multi-scale feature weighting trong YOLO. Phương pháp dùng baseline detector đã train để khai thác object-scale activation evidence trên các feature level `P2/P3/P4/P5`, tính các evidence metrics như in-box energy, peak alignment và background leakage, hiệu chỉnh hoặc stress-test các hệ số evidence score, tổng hợp scale preference theo nhóm kích thước object, rồi sinh runtime policy bảo thủ với identity fallback khi evidence không đủ chắc chắn. Để xử lý mismatch giữa GT box ở mức instance và feature map ở mức image, phương pháp áp policy thông qua instance-aware spatial masks: mỗi GT object được project xuống các feature map tương ứng, scale weighting chỉ được áp bên trong vùng không gian của object, còn background và vùng không chắc chắn giữ identity. Toàn bộ workflow được triển khai bằng một guarded all-in-one Python entrypoint, mặc định non-execution, có artifact logging, runtime validation, và chỉ train khi được bật rõ ràng bằng `--stage train --execute`.

---

## 30. Implementation checklist

```text
[x] Tạo scripts/run_scale_pipeline.py
[ ] Thêm CLI args và stage control
[ ] Implement baseline loading
[ ] Implement feature scale resolution
[ ] Implement activation hooks
[ ] Implement YOLO label parsing
[ ] Implement object size grouping
[ ] Implement object-scale metric extraction
[ ] Implement evidence score computation
[ ] Implement coefficient sensitivity analysis
[ ] Implement aggregation by size_group x scale
[ ] Implement policy candidate mining
[ ] Implement guardrails và identity fallback
[ ] Implement runtime policy export
[ ] Implement runtime policy schema validation
[ ] Implement projected GT spatial masks
[ ] Implement instance-aware feature weighting
[ ] Implement dry-run forward compatibility check
[ ] Implement non-execution manifest
[ ] Implement optional train execution guard
[ ] Implement report và README artifact writing
[x] Đồng bộ tài liệu hiện có trong docs/m8_note.md
[ ] Add experiment_log entry
```
