import mlflow
import mlflow.sklearn

def log_experiment(model_name, model, params, metrics, X_test, y_test):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("stage", "development")
        print(f"Logged model: {model_name}")
