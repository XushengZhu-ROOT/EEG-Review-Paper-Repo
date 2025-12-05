See `scripts/review-finetune-<dataset>.sh` to check finetuning experiment setting

Before finetuning, you need to use generate `tMatrix_value.npy` and put in `./inputs` folders. Remember to modify codes of dataset part in `src/train_gpt.py`

## Data process
- bci2a (NeuroGPT author):
    - sample rate = 250
    - divide 4 seconds data into 2 chunk, 2 seconds per chunk with no overlap sample points.
- Stress:
    - sample rate = 200
    - divide 5 seconds data into 2 chunk, 2.5 seconds per chunk with no overlap sample points.
- KaggleERN:
    - sample rate = 250
    - divide 3 seconds data into 2 chunk, 2 seconds per chunk with 250 overlaped sample points.