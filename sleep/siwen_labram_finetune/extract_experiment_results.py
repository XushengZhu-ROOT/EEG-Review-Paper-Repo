import os
import json
import re
import pandas as pd

def extract_results():
    base_dir = "checkpoints/Sleep_search-noPosWeight"
    
    # Regex to parse folder name
    # Pattern: swien_config_hpo_exp${exp_count}_lr${lr}_wd${wd}_bs${bs}-1ly-all
    # Example: swien_config_hpo_exp1_lr0.001_wd0.01_bs256-1ly-all
    folder_pattern = re.compile(r"swien_config_hpo_exp(\d+)_lr([\d\.]+)_wd([\d\.]+)_bs(\d+)-1ly-all")
    
    results = []
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    for folder_name in os.listdir(base_dir):
        match = folder_pattern.match(folder_name)
        if match:
            exp_count = int(match.group(1))
            lr = float(match.group(2))
            wd = float(match.group(3))
            bs = int(match.group(4))
            
            log_path = os.path.join(base_dir, folder_name, "log.txt")
            
            if not os.path.exists(log_path):
                print(f"Log file not found for {folder_name}")
                continue
                
            best_val_bacc = -1.0
            best_test_bacc = -1.0
            best_epoch = -1
            
            try:
                with open(log_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            val_bacc = data.get('val_balanced_accuracy', -1)
                            test_bacc = data.get('test_balanced_accuracy', -1)
                            epoch = data.get('epoch', -1)
                            
                            if val_bacc > best_val_bacc:
                                best_val_bacc = val_bacc
                                best_test_bacc = test_bacc
                                best_epoch = epoch
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error reading {log_path}: {e}")
                continue
            
            results.append({
                "exp_count": exp_count,
                "lr": lr,
                "wd": wd,
                "bs": bs,
                "best_val_bacc": best_val_bacc,
                "test_bacc": best_test_bacc,
                "best_epoch": best_epoch
            })

    # Sort results by experiment count
    results.sort(key=lambda x: x["exp_count"])
    
    # Print header
    print(f"{'Exp':<5} {'LR':<10} {'WD':<10} {'BS':<5} {'Best Val BAcc':<15} {'Test BAcc':<15} {'Best Epoch':<10}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['exp_count']:<5} {res['lr']:<10} {res['wd']:<10} {res['bs']:<5} {res['best_val_bacc']:<15.4f} {res['test_bacc']:<15.4f} {res['best_epoch']:<10}")
    
    # Convert results to DataFrame for better visualization or further processing
    df = pd.DataFrame(results)
    print("\nDataFrame representation:")
    print(df)
    df.to_csv("experiment_results.csv", index=False)

if __name__ == "__main__":
    extract_results()
