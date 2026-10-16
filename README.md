# Machine Learning from Scratch

This is a Python project I'm building to understand the fundamentals of machine learning by implementing the algorithms and supporting tools myself. The current version includes a binary perceptron, feature scaling, a train/test split, and classification evaluation.

**Details:** Python 3.10+, NumPy, pandas, and Matplotlib for plotting  
**Status:** In development  
**Current model:** Single-layer perceptron

## Examples

### Loan-applicant classification

[loan_applicants.py](loan_applicants/loan_applicants.py) brings the different parts of the project together using a CSV containing 1,000 generated practice samples.

The features are annual income, credit score, and debt ratio, with an approved/not-approved label as the target. The dataset was generated with GPT-6 for this project.

The example:

1. Loads the CSV into a pandas DataFrame.
2. Shuffles and splits the features and labels into training and test sets.
3. Fits the scaler on the training features and uses those same statistics to transform the test features.
4. Trains the perceptron by updating its weights and bias when it makes an incorrect prediction.
5. Predicts labels for the test set and prints accuracy, precision, recall, and a confusion matrix.

The split and training settings are defined in [loan_applicants.py](loan_applicants/loan_applicants.py). I change these while experimenting with training and evaluation, so the results depend on the chosen settings.

### Logical functions and decision boundaries

[logical_functions.py](logical_functions.py) trains the perceptron on a small truth table and plots its learned decision boundary with Matplotlib.

The example uses AND by default. Changing `selected_targets` lets me try OR, XOR, and XNOR using the target arrays already defined in the script.

AND and OR can be separated with a straight line. XOR and XNOR demonstrate the limitation of using a single linear decision boundary, which is part of why I want to build toward a multilayer perceptron.

## Running Locally

Run the following commands from the repository's root folder. They create a virtual environment and install the project with its plotting dependency.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[plots]"
```

Run the loan-applicant example:

```powershell
.\.venv\Scripts\python.exe loan_applicants\loan_applicants.py
```

Run the logical-function plot:

```powershell
.\.venv\Scripts\python.exe logical_functions.py
```

These commands use the virtual environment's Python directly, so activating the environment is not required.

### macOS or Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[plots]"
.venv/bin/python loan_applicants/loan_applicants.py
.venv/bin/python logical_functions.py
```

Matplotlib is optional for the library and loan-applicant example. To install without it, use `python -m pip install -e .` with your chosen Python environment.

## Code Tour

These files are good starting points for looking through the project:

| Starting point | What to look for |
| --- | --- |
| [perceptron.py](connor_ml/perceptron.py) | Initializes the weights and bias, makes predictions using a weighted sum and threshold, and updates the parameters when a training prediction is incorrect. Training stops when an entire epoch has no mistakes or the maximum epoch count is reached. |
| [scaler.py](connor_ml/scaler.py) | Calculates and stores each feature's mean and sample standard deviation, then reuses those statistics when standardizing new data. |
| [train_selection.py](connor_ml/train_selection.py) | Uses a seeded random permutation to shuffle row positions while keeping features and labels paired, then separates the training and test sets. |
| [evaluator.py](connor_ml/evaluator.py) | Calculates accuracy, precision, recall, and a confusion matrix with totals. Precision, recall, and the matrix use a specified positive class. |
| [loan_applicants.py](loan_applicants/loan_applicants.py) | Connects loading, splitting, scaling, training, prediction, and evaluation into one runnable example. |
| [logical_functions.py](logical_functions.py) | Uses truth tables and a plotted decision boundary to make the perceptron's behavior easier to inspect. |

## Looking Back

I started this project because I felt like I was falling behind in my machine learning class. There were foundational concepts I didn't feel comfortable with yet, so I decided to try implementing things from scratch to force myself to understand what was actually happening.

One thing that's helped is being able to recognize the bigger pieces and why certain things need to happen in a particular order. For example, splitting the data before fitting the scaler makes more sense to me now because I've worked through how the training and test data are used.

There's still a lot for me to learn, but this project has given me something I can experiment with as I go. I've even intentionally changed the training settings to produce more mistakes so I could check my evaluation metrics and better understand what the results were telling me.

I plan to keep building on this as I learn more, with a multilayer perceptron, gradient-based training, and backpropagation as my next goals.

## Learning Resources

[learning-sources.md](learning-sources.md) lists the perceptron resources I've been following.
