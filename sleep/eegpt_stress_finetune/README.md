## ⚙️ Usage

### 1. **Set the visible GPUs**

Edit the following line near the top of the script to choose which GPUs to use:

```python
GPU_VISIBLE = "3,4,5"
```

### 2. **Download the pretrained EEGPT checkpoint**

Download the EEGPT pretraining checkpoint here:
👉 [EEGPT Checkpoint](https://figshare.com/s/e37df4f8a907a866df4b)

Save the file in this directory **or** update the checkpoint path inside the script:

```python
ckpt_path: str = "./eegpt_mcae_58chs_4s_large4E.ckpt"
```

### 3. **Run fine-tuning**

#### **Linear probe (head-only fine-tuning; encoder frozen)**

```bash
python3 finetune_linear_probe.py
```

#### **Full fine-tuning (unfreeze and train the whole model)**

```bash
python3 finetune_all.py
```

All training logs and best checkpoints will be saved under:

```
./outputs_eegpt_stress/
```
