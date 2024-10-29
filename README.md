# ttc
Do DDMs show effective test-time-compute performance?

### $4 \times 4$, $5 \times 5$ multiplication

```bash
# dot (T=64 by default)
weights_path="weights_path"
python3 evaluation_batch.py      \
    --weights_path $weights_path \
    --fix_src               \
    --digit                 \
    --dataset gsm8k         \
    --score_temp 0.5        \
    --sampling_timesteps 8
```