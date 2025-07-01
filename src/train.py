import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from src.utils import evaluate_model
from src.mlflow_utils import log_experiment

# Load preprocessed features and labels
X = joblib.load("data/X_features.pkl")
y = joblib.load("data/y_labels.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models and grids
models = {
    "logistic_regression": (LogisticRegression(max_iter=1000), {
        "C": [0.01, 0.1, 1.0],
        "solver": ["liblinear"]
    }),
    "random_forest": (RandomForestClassifier(), {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 10]
    }),
    "decision_tree": (DecisionTreeClassifier(), {
        "max_depth": [3, 5, 10]
    }),
    "gbm": (GradientBoostingClassifier(), {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1]
    })
}

for model_name, (model, param_grid) in models.items():
    print(f"\nTraining: {model_name}")
    grid = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_proba)
    print(f"Metrics: {metrics}")

    log_experiment(model_name, best_model, grid.best_params_, metrics, X_test, y_test)

    # Optional: Save best model
    joblib.dump(best_model, f"models/{model_name}_best_model.pkl")
# Save processed data
import joblib
def save_processed_data(df, save_path):
    """
    Save processed DataFrame to a specified path.
    """
    df.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")
    