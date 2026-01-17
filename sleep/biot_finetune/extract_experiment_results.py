import os
import re
import pandas as pd
import tensorflow as tf

try:
    import yaml
except ImportError:  # fallback if PyYAML is unavailable
    yaml = None


DATASET_TRAIN_DIRS = {
    "ISRUC": "/home/dung/Documents/EEG-Review-Paper-Repo/preprocessing/data/isruc_biot/train",
}

_DATASET_STEP_CACHE = {}


def get_steps_per_epoch(dataset, batch_size):
    """Return steps per epoch for a dataset given batch size (drop_last training)."""
    if not dataset or batch_size <= 0:
        return None
    cache_key = (dataset, batch_size)
    if cache_key in _DATASET_STEP_CACHE:
        return _DATASET_STEP_CACHE[cache_key]

    train_dir = DATASET_TRAIN_DIRS.get(dataset)
    if not train_dir or not os.path.exists(train_dir):
        return None

    try:
        num_samples = sum(
            1
            for entry in os.scandir(train_dir)
            if entry.is_file()
        )
    except OSError as exc:
        print(f"Warning: could not count samples for {dataset}: {exc}")
        return None

    steps = num_samples // batch_size
    if steps == 0:
        return None
    _DATASET_STEP_CACHE[cache_key] = steps
    return steps


def get_scalar_run_tensorboard(tag, filepath):
    """Read scalar events for a tag and keep (step, value) pairs."""
    values = []
    try:
        for event in tf.compat.v1.train.summary_iterator(filepath):
            if not event.summary.value:
                continue
            for summary_value in event.summary.value:
                if summary_value.tag != tag:
                    continue
                if summary_value.HasField('simple_value'):
                    value = summary_value.simple_value
                elif summary_value.HasField('tensor'):
                    value = tf.make_ndarray(summary_value.tensor).item()
                else:
                    continue
                values.append({"step": event.step, "value": value})
    except Exception as exc:
        print(f"Error reading {filepath}: {exc}")
    return values

def extract_results():
    base_dir = "/home/dung/Documents/EEG-Review-Paper-Repo/biot_finetune/log/ISRUC-posWeight-rerun"
    
    # Regex to parse folder name
    # Pattern: exp_hpo1-CBraMod_3lyStyle_LayerNorm-BIOT_3ly--lr0.005-bs256-wd0.001-sr200-ts200-hl100
    folder_pattern = re.compile(r"exp_hpo(\d+)-LabramClassifier-BIOT_3ly-all-lr([\d\.e-]+)-bs(\d+)-wd([\d\.e-]+)-sr(\d+)-ts(\d+)-hl(\d+)")
    
    results = []
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    for folder_name in os.listdir(base_dir):
        match = folder_pattern.match(folder_name)
        if match:
            exp_count = int(match.group(1))
            lr = float(match.group(2))
            bs = int(match.group(3))
            wd = float(match.group(4))
            sr = int(match.group(5))
            ts = int(match.group(6))
            hl = int(match.group(7))
            
            folder_path = os.path.join(base_dir, folder_name)
            
            best_val_bacc = None
            best_test_bacc = None
            best_step = None
            converted_epoch = None

            dataset_name = None
            epochs_config = None
            config_path = os.path.join(folder_path, "config.yaml")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as cfg_file:
                        cfg = yaml.safe_load(cfg_file)
                        dataset_name = cfg.get("dataset")
                        epochs_config = cfg.get("epochs")
                except yaml.YAMLError as exc:
                    print(f"Warning: could not parse {config_path}: {exc}")
            
            # Find specific event files
            val_file = next((f for f in os.listdir(folder_path) if "events.out.tfevents" in f and f.endswith(".0")), None)
            test_file = next((f for f in os.listdir(folder_path) if "events.out.tfevents" in f and f.endswith(".1")), None)

            if val_file:
                val_path = os.path.join(folder_path, val_file)
                val_values = get_scalar_run_tensorboard("val_balanced_acc", val_path)
                if val_values:
                    best_entry = max(val_values, key=lambda item: item["value"])
                    best_val_bacc = best_entry["value"]
                    best_step = best_entry["step"]
            
            if test_file:
                test_path = os.path.join(folder_path, test_file)
                test_values = get_scalar_run_tensorboard("test_balanced_acc", test_path)
                if test_values:
                    if best_step is not None:
                        test_by_step = {item["step"]: item["value"] for item in test_values}
                        best_test_bacc = test_by_step.get(best_step)
                    if best_test_bacc is None:
                        best_test_bacc = max(test_values, key=lambda item: item["value"])["value"]

            steps_per_epoch = get_steps_per_epoch(dataset_name, bs)
            if best_step is not None and steps_per_epoch:
                converted_epoch = int(best_step // steps_per_epoch) + 1
                if isinstance(epochs_config, (int, float)):
                    converted_epoch = min(converted_epoch, int(epochs_config))
            
            results.append({
                "exp_count": exp_count,
                "lr": lr,
                "bs": bs,
                "wd": wd,
                "sr": sr,
                "ts": ts,
                "hl": hl,
                "best_val_bacc": best_val_bacc,
                "best_test_bacc": best_test_bacc,
                "best_epoch": converted_epoch,
                "best_step": best_step
            })

    # Sort results by experiment count
    results.sort(key=lambda x: x["exp_count"])
    
    # Print header
    print(f"{'Exp':<5} {'LR':<10} {'BS':<5} {'WD':<10} {'SR':<5} {'TS':<5} {'HL':<5} {'Best Val BAcc':<15} {'Best Test BAcc':<15} {'Best Epoch':<12} {'Best Step':<10}")
    print("-" * 135)
    
    for res in results:
        val_str = f"{res['best_val_bacc']:.4f}" if res['best_val_bacc'] is not None else "N/A"
        test_str = f"{res['best_test_bacc']:.4f}" if res['best_test_bacc'] is not None else "N/A"
        epoch_str = f"{int(res['best_epoch'])}" if res['best_epoch'] is not None else "N/A"
        step_str = f"{int(res['best_step'])}" if res['best_step'] is not None else "N/A"
        print(f"{res['exp_count']:<5} {res['lr']:<10} {res['bs']:<5} {res['wd']:<10} {res['sr']:<5} {res['ts']:<5} {res['hl']:<5} {val_str:<15} {test_str:<15} {epoch_str:<12} {step_str:<10}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    print("\nDataFrame representation:")
    print(df)
    df.to_csv("experiment_results.csv", index=False)

if __name__ == "__main__":
    extract_results()