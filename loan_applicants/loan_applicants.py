from pathlib import Path
import pandas as pd

from connor_ml.train_selection import train_test_split
from connor_ml.perceptron import Perceptron
from connor_ml.scaler import Scaler
from connor_ml.evaluator import Evaluator

CSV_FILE = "ml_loan_applicants.csv"
RANDOM_STATE = 33

# Get the .csv file to be read
file_path = Path(__file__).parent / CSV_FILE
df = pd.read_csv(file_path) # this always returns a DataFrame so it can just be directly assigned

# Create the features and target data
X = df.drop(columns=["approved"])
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

scaler = Scaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

perceptron = Perceptron(max_epochs=100, learning_rate=0.1, random_state=RANDOM_STATE)
perceptron.fit(X_train_standardized, y_train)

y_pred = perceptron.predict(X_test_standardized)

evaluator = Evaluator(y_pred, y_test)
accuracy = evaluator.get_accuracy()
print("Accuracy:", accuracy)

precision = evaluator.get_precision(1)
print("Precision:", precision)