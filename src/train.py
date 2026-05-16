import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    train()