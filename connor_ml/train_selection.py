import numpy as np
import pandas as pd

def train_test_split(X: pd.DataFrame | pd.Series, y: pd.Series, test_size: float=0.25, random_state: int=42):
    '''Split data between training data and test data.

    Args:
        X: Feature data (expected to be a 2D pandas DataFrame or a 1D pandas Series).
        y: Targets (target labels, expected to be a 1D pandas Series).
        test_size: What percent (as a decimal) to keep for testing (e.g., test_size=0.3 means 30% of the data
        is used for testing). Defaults to 0.25.
        random_state: Set the seed for the shuffling of data, as the data is shuffled before it's split between
        the train and test structures.

    Returns:
        (X_train, X_test, y_train, y_test). The returned objects perserve their respective pandas types.
    '''

    length = len(X)

    if length != len(y):
        raise ValueError("X and y must contain the same number of samples!")

    # Generates a randomized ordering of row positions
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(length)

    # Apply the randomly generated ordering to both features and targets
    shuffled_X = X.iloc[indices].reset_index(drop=True)
    shuffled_y = y.iloc[indices].reset_index(drop=True)

    # If X has 1000 rows and test_size=0.3, use 700 rows for training and keep 300 for testing
    train_count = int(length * (1 - test_size))
    test_count = length - train_count

    X_train = shuffled_X.iloc[:train_count]
    y_train = shuffled_y.iloc[:train_count]

    test_index_slice = (train_count + test_count)
    X_test = shuffled_X.iloc[train_count:test_index_slice]
    y_test = shuffled_y.iloc[train_count:test_index_slice]

    return (X_train, X_test, y_train, y_test)
