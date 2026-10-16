import pandas as pd

# NOTE: Scaling statistics should never be calculated using the test set because then the test results leak into
# the training data, so fit_transform() should be used for training data and transform() should be used for test data

class Scaler:
    '''Standardize numerical features using Z-score scaling.

    Each feature is standardized by subtracting its mean and dividing by its standard deviation. The mean and
    standard deviation are calculated separately for each column.

    The scaler stores these statistics so they can be reused when transforming new data without recalculating them.
    '''

    def __init__(self):
        # Means of each column are stored in a pd.Series or as a scalar if the input itself is a pd.Series
        self.means_ = None
        # Standard deviations of each column are stored in a pd.Series or as a scalar if the input itself is a pd.Series
        self.std_devs_ = None

    def fit(self, X: pd.DataFrame | pd.Series):
        '''Calculate and store scaling statistics from the input data.

        Calculates the mean and sample standard deviation (ddof=1) for each feature. The input data is not modified
        or transformed.

        Args:
            X: A pandas DataFrame or Series containing numerical features. For a DataFrame, each row represents a
            sample and each column a feature where statistics are calculated separately for each column, so a DF
            with 3 columns will have 3 means and 3 standard deviations stored in self.means_ and self.std_devs_.
            For a Series, one mean and standard deviation are calculated.

        Returns:
            None. The calculated statistics are stored in self.means_ and self.std_devs_.
        '''
        self.means_ = X.mean(axis=0)
        self.std_devs_ = X.std(axis=0, ddof=1)

    def transform(self, X: pd.DataFrame | pd.Series):
        '''Standardize data using the previously calculated statistics.

        Does not recalculate the statistics or modify the original DataFrame or Series. The scaler must be fitted before
        calling this method.

        Args:
            X: A pandas DataFrame or Series containing numerical features with the same columns as the fitted data.

        Returns:
            A new pandas DataFrame or Series containing the standardized values.

        Raises:
            RuntimeError: If the scaler has not been fitted.
        '''
        if self.means_ is None or self.std_devs_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform(). Use fit() or fit_transform().")

        if self.std_devs_.eq(0).any():
            raise ValueError("Standard deviation is zero for one or more features. Cannot standardize.")
        
        X_standardized_z = (X - self.means_) / self.std_devs_
        return X_standardized_z

    def fit_transform(self, X: pd.DataFrame | pd.Series):
        '''Calculate scaling statistics and standardize the input data.

        Calls fit() to calculate and store the mean and standard deviation for each feature, then calls transform()
        to standardize the data using those statistics.

        Args:
            X: A pandas DataFrame or Series containing numerical features.

        Returns:
            A new pandas DataFrame or Series containing the standardized values.
        '''
        self.fit(X)
        return self.transform(X)
