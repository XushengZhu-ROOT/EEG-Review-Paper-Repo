## ⚙️ Usage

### 1. **Set the visible GPUs**
Edit the following line near the top of the script to specify your GPU IDs:
```python
GPU_VISIBLE = "3,4,5"
```

### 2. **Download the pretrained EEGPT checkpoint**

Download the EEGPT pretraining checkpoint here:
👉 [EEGPT Checkpoint]([https://your-checkpoint-link-here](https://figshare.com/s/e37df4f8a907a866df4b))

After downloading, place it in this folder or update the path inside the script:

```python
ckpt_path: str = "./eegpt_mcae_58chs_4s_large4E.ckpt"
```

### 3. **Run fine-tuning**

```bash
python3 large_d_run_class_finetuning.py
```

Training logs and best model checkpoints will be automatically saved under:

```
./outputs_eegpt_stress/
```
