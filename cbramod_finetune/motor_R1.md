**Context & Current Status:**
I have a PyTorch based classification pipeline for a multiactivity IMU EEG dataset. Currently the code takes one second non overlapping EEG epochs from 20 subjects and randomly divides them into a training set, a validation set, and a test set. The model fine tunes a pretrained CBraMod backbone network for a six class classification task.

**The Problem:**
Peer reviewers pointed out that a random epoch split causes severe data leakage because adjacent one second windows from the same continuous recording are highly autocorrelated. We need to implement a Subject Independent evaluation strategy.

**Your Task:**
Refactor the dataset splitting and training evaluation loops to implement a Subject Independent validation strategy. Since I want to test the code logic quickly, I only want to run a single fold dry run right now, rather than a full Leave One Subject Out loop.

**Specific Technical Requirements:**

* **Subject Independent Data Split:** Modify the PyTorch Dataset or DataLoader logic to group data strictly by Subject ID. You must create three strictly disjoint sets: 18 subjects for the Training set, 1 subject for the Validation set (to monitor loss and select the best epoch), and 1 subject for the Test set.
* **Single Fold Dry Run:** Implement a boolean flag `single_fold_debug = True`. When this flag is active, the training loop should only execute the very first fold (for example, training on subjects 1 to 18, validating on subject 19, and testing on subject 20) and then stop completely.
* **Expanded Metrics:** Update the evaluation function using scikit learn metrics to compute and log the following for the Test set at the end of the run:
* Balanced Accuracy (Arithmetic mean of the recall across all six classes).
* Macro F1 Score.
* Per class Recall.
* Confusion Matrix.


* **Random Seed:** Ensure reproducibility by fixing the random seeds for numpy and torch.

Please analyze my current classification code and provide the refactored code blocks to achieve this subject independent single fold training pipeline.