# Experiment Log

## 2026-06-30
- Phase: `M8.1`
- Script: `scripts/prepare_m8_1_size_aware_fpn_scale_selection_decision_gate.py`
- Output dir: `artifacts/m8_1_size_aware_fpn_scale_selection_decision_gate`
- Status: `m8_1_decision_gate_prepared`
- Selected branch: `m8_v1a_conservative_static_scale_weighting`
- Execution guardrail: no training, no evaluation, no inference, no dataset mutation, no checkpoint mutation

- Phase: `M8_v1b`
- Scripts: `scripts/prepare_m8_v1b_size_aware_scale_weighting_implementation.py`, `scripts/check_m8_v1b_size_aware_scale_weighting.py`
- Output dir: `artifacts/m8_v1b_size_aware_scale_weighting_implementation`
- Status: `m8_v1b_size_aware_scale_weighting_implementation_prepared`
- Human override: `m8_v1a_conservative_static_scale_weighting -> m8_v1b_size_aware_scale_weighting`
- Execution guardrail: no training, no evaluation, no inference, no dataset mutation, no checkpoint mutation

- Phase: `M8_v1b.1`
- Scripts: `scripts/prepare_m8_v1b_1_policy_alignment_patch.py`, `scripts/check_m8_v1b_size_aware_scale_weighting.py`
- Output dir: `artifacts/m8_v1b_1_policy_alignment_patch`
- Status: `m8_v1b_1_policy_alignment_patch_prepared`
- Evidence alignment: `small -> P3`, `medium -> P5`, `large -> P5`, `unknown -> identity`
- Execution guardrail: no training, no evaluation, no inference, no dataset mutation, no checkpoint mutation
