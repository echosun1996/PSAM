# Comparison: main.sh vs point_ablation_study.sh

## Key Differences in train.py Evaluation Calls

### main.sh (line 511)
```bash
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nnodes=1 --nproc_per_node=1 \
  train.py --eval \
  --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --output $parent_dir/results/PSAM \
  --restore-model $parent_dir/results/PSAM/epoch_16.pth \
  --input $parent_dir/data \
  --input_size $img_resize $img_resize \
  --logfile $parent_dir/results/PSAM/PSAM_eva.txt \
  --visualize \
  --comments "$wandb_comment" \
  --labeller $labeller
```

**Missing parameters (using defaults):**
- `--n-sample-points`: default=4
- `--n-positive-points`: default=4
- `--n-negative-points`: default=4
- `--prompt_type`: default="P_B"
- `--token-visualisation`: default=False
- `--single-image`: default=None (processes ALL images)

### point_ablation_study.sh (lines 74-89)
```bash
CUDA_VISIBLE_DEVICES=$gpu_id $torchrun_cmd --nnodes=1 --nproc_per_node=$gpu_sum \
  train.py --eval \
  --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --output $output_dir \
  --restore-model $parent_dir/results/PSAM/epoch_16.pth \
  --input $parent_dir/data \
  --input_size $img_resize $img_resize \
  --logfile $output_dir/PSAM_eva.txt \
  --visualize \
  --comments "Point ablation: n-neg=4, n-pos=$n_pos" \
  --labeller $labeller \
  --token-visualisation True \
  --n-positive-points $n_pos \
  --n-negative-points 4 \
  --single-image "$image_filename"
```

**Explicitly specified parameters:**
- `--token-visualisation True` ⚠️ **CRITICAL DIFFERENCE**
- `--n-positive-points $n_pos` (varies: 3-5)
- `--n-negative-points 4`
- `--single-image "$image_filename"` ⚠️ **CRITICAL DIFFERENCE**
- `--nproc_per_node=$gpu_sum` (uses all GPUs vs single GPU)

## Critical Differences That Could Cause Inference Differences

1. **`--token-visualisation True`**: 
   - `point_ablation_study.sh` enables this, `main.sh` does not
   - This generates additional visualization files (`_ps.png`, `_ns.png`, `_uncertain.png`)
   - **May affect inference behavior** if the code path for token visualization modifies the inference logic

2. **`--single-image`**:
   - `point_ablation_study.sh` processes only ONE image
   - `main.sh` processes ALL validation images
   - This shouldn't affect inference results for the same image, but could affect batch processing

3. **GPU Configuration**:
   - `main.sh`: `--nproc_per_node=1` (single GPU)
   - `point_ablation_study.sh`: `--nproc_per_node=$gpu_sum` (multiple GPUs)
   - **Could cause non-deterministic results** if there are race conditions or different random seeds per GPU

4. **`--n-positive-points` and `--n-negative-points`**:
   - `point_ablation_study.sh` explicitly sets these (varies by scenario)
   - `main.sh` uses defaults (both = 4)
   - **This WILL cause different results** if comparing scenarios with n_pos != 4 or n_neg != 4

5. **`--prompt_type`**:
   - `point_ablation_study.sh` explicitly sets this in Scenarios 3-5 (P_B, P, B)
   - `main.sh` uses default "P_B"
   - **This WILL cause different results** for Scenarios 4 and 5

## Recommendations for point_ablation_study.sh

To ensure consistency with `main.sh` when comparing the same image:

1. **Check if `--token-visualisation` affects inference**: Review `train.py` to see if enabling token visualization changes the inference logic or only adds visualization outputs.

2. **Ensure deterministic GPU behavior**: Consider using `--nproc_per_node=1` or setting a fixed random seed to ensure reproducibility.

3. **Verify parameter defaults**: When comparing with `main.sh`, ensure that `point_ablation_study.sh` explicitly sets all parameters that `main.sh` uses as defaults, OR document that the differences are intentional for ablation studies.

4. **Check for batch processing differences**: The `--single-image` parameter might affect how the model processes data (e.g., batch normalization behavior).






