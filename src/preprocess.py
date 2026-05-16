import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Renombrar variable objetivo
    if "Machine failure" in df.columns:
        df = df.rename(columns={
            "Machine failure": "Target"
        })

    return df


def prepare_data(df: pd.DataFrame):
    df = df.copy()

    # Eliminar columnas innecesarias
    df = df.drop(
        columns=[
            "UDI",
            "Product ID",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF"
        ],
        errors="ignore"
    )

    # Separar variables
    X = df.drop(columns=["Target"])
    y = df["Target"]

    return X, y