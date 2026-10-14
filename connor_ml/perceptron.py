import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, max_epochs: int, learning_rate: float, random_state: int=None):
        '''Initialize a perceptron object.
        
        Args:
            max_epochs: The maximum number of passes of the training data.
            learning_rate: A constant by which the weights/bias updates are scaled during training.
            random_state: Optional seed that can be used to randomly initialize the starting weights.
        '''
        # NOTE: even if random_state stays None, np.random.default_rng(self.random_state) automatically will
        # use nondeterministic initialization so a random number is selected without any seed

        self.max_epochs = max_epochs # sklearn calls this max_iter but it's the same thing
        self.learning_rate = learning_rate
        self.random_state = random_state
    
    def fit(self, X_train: pd.DataFrame | pd.Series | np.ndarray, y_train: pd.Series | np.ndarray, debug: bool=False):
        '''Train the perceptron using the provided features and targets labels.

        Training continues for a maximum of max_epochs passes through the dataset, or stops early if an entire
        epoch has no errors.

        Args:
            X_train: Training features as a pandas DataFrame, Series, or NumPy array.
            y_train: Correct binary target labels (0 or 1) for each sample, stored as a pandas Series or NumPy array.
            debug: If set to true, print() statements will be used to log steps in the training process.

        Returns:
            None. The learned weights and bias are stored in self.w_ and self.b_.
        '''

        # Convert X_train/y_train to numpy array to work with zip() as is
        X = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series) else X_train
        y = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        # Initialize random small weights
        rng = np.random.default_rng(self.random_state)
        num_of_features = X.shape[1] if X.ndim > 1 else 1 # One feature for 1D inputs, otherwise one weight per column
        self.w_ = rng.uniform(low=-0.1, high=0.1, size=num_of_features)

        if debug:
            print("\nStarting weights:", self.w_, "\n")

        # Bias starts as -1 but can change during training, so bias is also a learnable parameter. The bias is
        # treated like a weight multiplied by a constant input of 1:
        # z = w1x1 + w2x2 + w3x3 + b(1), where b * (1) is the bias
        self.b_ = -1

        for epoch in range(self.max_epochs):
            error_count = 0
            for row, target in zip(X, y):
                y_pred = self.predict(row)
                error = y_pred - target
                if error != 0.0:
                    # Update the weights (numpy handles array subtraction, python can't do this natively)
                    self.w_ = self.w_ - self.learning_rate * error * row
                    # Update the bias
                    self.b_ = self.b_ - self.learning_rate * error
                    error_count += 1
                    if debug:
                        print(f"Epoch {epoch}: Row {row} predicting {y_pred} when target = {target}. Updating weights: {self.w_}. Updating bias: {self.b_}")
            if error_count == 0:
                if debug:
                    print(f"\nNo more errors after {epoch} epochs!\n")
                break

    def predict(self, X: pd.DataFrame | pd.Series | np.ndarray):
        '''Predict class labels using the perceptron's current weights and bias.

        Args:
            X: Input features. Can be a single sample (1D array) or multiple samples (2D array).

        Returns:
            The predicted class label (0 or 1) for a single sample, or a NumPy array of labels for multiple samples.
        '''

        weighted_sum = np.dot(X, self.w_) + self.b_
        return np.where(weighted_sum >= 0.0, 1, 0)
