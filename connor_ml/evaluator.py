import numpy as np
import pandas as pd

class Evaluator:
    def __init__(self, y_pred: pd.Series | np.ndarray, targets: pd.Series | np.ndarray):
        self.set(y_pred, targets)

    def set(self, y_pred: pd.Series | np.ndarray, targets: pd.Series | np.ndarray):
        self.y_ = y_pred.to_numpy() if isinstance(y_pred, pd.Series) else y_pred
        self.t_ = targets.to_numpy() if isinstance(targets, pd.Series) else targets
        self.total_ = len(targets)

    def get_accuracy(self):
        '''Calculate the accuracy of the y_pred and targets set via the constructor or set().

        Accuracy is calculated as: (TP + TN) / Total, where TP = number of true positives, TN = number of true
        negatives, and Total = the total size of the targets/prediction Series or array.

        Returns:
            The calculated accuracy of the prediction.
        '''
        # Creates an array of zeros and non-zeros where a zero element is a correct prediction and a non-zero
        # element is an incorrect prediction (e.g., if the target was true (1) but the prediction was false (0),
        # 1 - 0 = 1 meaning the prediction was a false negative
        results = self.y_ - self.t_
        correct_results = results[results == 0] # Use a mask to get only the correct predictions
        return len(correct_results) / len(results) # Divide total number of correct predictions (TP + TN) by total

    def get_precision(y_pred: pd.Series | np.ndarray, targets: pd.Series | np.ndarray):
        return 0

    def get_recall(y_pred: pd.Series | np.ndarray, targets: pd.Series | np.ndarray):
        return 0
