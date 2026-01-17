import os
import json
import re
import pandas as pd

def extract_results():
    base_dir = "checkpoints/ISRUC"
    
    # Regex to parse folder name
    # TODO: update regex for different experiment type
    folder_pattern = re.compile(r"hpo_exp(\d+)_lr([\d\.]+)_wd([\d\.]+)_bs(\d+)-all_patch_reps-all")
    
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
            
            summary_path = os.path.join(base_dir, folder_name, "training_summary.json")
            log_path = os.path.join(base_dir, folder_name, "log.txt")
            
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    
                    final_results = summary.get('final_results', {})
                    training_info = summary.get('training_info', {})
                    gpu_stats = summary.get('gpu_stats', {}).get('gpu_0', {})
                    
                    results.append({
                        "exp_count": exp_count,
                        "lr": lr,
                        "wd": wd,
                        "bs": bs,
                        "best_epoch": final_results.get('best_epoch'),
                        "val_bacc": final_results.get('val_bacc'),
                        "test_bacc": final_results.get('test_bacc'),
                        # "val_acc": final_results.get('val_acc'),
                        # "test_acc": final_results.get('test_acc'),
                        # "val_kappa": final_results.get('val_kappa'),
                        # "test_kappa": final_results.get('test_kappa'),
                        # "val_f1": final_results.get('val_f1'),
                        # "test_f1": final_results.get('test_f1'),
                        # "total_time": training_info.get('total_training_time_seconds'),
                        # "avg_epoch_time": training_info.get('average_time_per_epoch_seconds'),
                        # "gpu_max_mem": gpu_stats.get('max_memory_allocated_GB')
                    })
                except Exception as e:
                    print(f"Error reading {summary_path}: {e}")
            else:
                print(f"No summary or log file found for {folder_name}")
                continue

    # Sort results by experiment count
    results.sort(key=lambda x: x["exp_count"])
    
    # Print header
    print(f"{'Exp':<5} {'LR':<10} {'WD':<10} {'BS':<5} {'Best Val BAcc':<15} {'Test BAcc':<15} {'Best Epoch':<10}")
    print("-" * 80)
    
    for res in results:
        val_bacc = res.get('val_bacc')
        test_bacc = res.get('test_bacc')
        best_epoch = res.get('best_epoch')
        
        val_bacc_str = f"{val_bacc:.4f}" if val_bacc is not None else "N/A"
        test_bacc_str = f"{test_bacc:.4f}" if test_bacc is not None else "N/A"
        best_epoch_str = f"{best_epoch}" if best_epoch is not None else "N/A"

        print(f"{res['exp_count']:<5} {res['lr']:<10} {res['wd']:<10} {res['bs']:<5} {val_bacc_str:<15} {test_bacc_str:<15} {best_epoch_str:<10}")
    
    # Convert results to DataFrame for better visualization or further processing
    df = pd.DataFrame(results)
    print("\nDataFrame representation:")
    print(df)
    df.to_csv("experiment_results.csv", index=False)

if __name__ == "__main__":
    extract_results()
