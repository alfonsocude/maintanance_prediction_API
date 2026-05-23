import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from xgboost import XGBClassifier

from src.preprocess import load_data, prepare_data


DATA_PATH = "data/ai4i2020.csv"
MODEL_PATH = "models/maintenance_model.pkl"


def train():
    df = load_data(DATA_PATH)
    X, y = prepare_data(df)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    xgb_params = {
        "colsample_bytree": 0.93543302348105,
        "gamma": 0.2050775601494591,
        "learning_rate": 0.17772747470459999,
        "max_depth": 5,
        "min_child_weight": 3,
        "n_estimators": 311,
        "reg_alpha": 0.010726091927327824,
        "reg_lambda": 1.0335996115869373,
        "scale_pos_weight": 41.1829268292683,
        "subsample": 0.8828920382577778,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42
    }

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**xgb_params))
        ]
    )

    # Entrenamiento final con todos los datos
    model.fit(X, y)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Modelo XGBoost guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    train()