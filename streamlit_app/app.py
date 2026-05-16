import sqlite3
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import shap
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"
MODEL_PATH = "models/maintenance_model.pkl"
DB_PATH = "predictions.db"


st.set_page_config(
    page_title="Predictive Maintenance Platform",
    page_icon="⚙️",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            machine_type TEXT,
            air_temperature REAL,
            process_temperature REAL,
            rotational_speed INTEGER,
            torque REAL,
            tool_wear INTEGER,
            prediction INTEGER,
            failure_expected INTEGER,
            probability REAL,
            risk_level TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(payload, result, risk_level):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            created_at,
            machine_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
            prediction,
            failure_expected,
            probability,
            risk_level
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        payload["Type"],
        payload["Air_temperature_K"],
        payload["Process_temperature_K"],
        payload["Rotational_speed_rpm"],
        payload["Torque_Nm"],
        payload["Tool_wear_min"],
        result["prediction"],
        int(result["failure_expected"]),
        result["probability"],
        risk_level
    ))
    conn.commit()
    conn.close()


def load_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df


def get_risk_level(probability):
    if probability < 0.50:
        return "Bajo"
    elif probability < 0.70:
        return "Medio"
    elif probability < 0.85:
        return "Alto"
    return "Crítico"


def create_gauge(probability, risk_level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": f"Nivel de riesgo: {risk_level}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 70], "color": "yellow"},
                {"range": [70, 85], "color": "orange"},
                {"range": [85, 100], "color": "red"},
            ],
        }
    ))
    fig.update_layout(height=320)
    return fig


def get_recommendations(input_df, probability, failure_expected):
    recommendations = []

    torque = input_df["Torque [Nm]"].iloc[0]
    rpm = input_df["Rotational speed [rpm]"].iloc[0]
    tool_wear = input_df["Tool wear [min]"].iloc[0]
    air_temp = input_df["Air temperature [K]"].iloc[0]
    process_temp = input_df["Process temperature [K]"].iloc[0]

    if failure_expected:
        recommendations.append("Activar protocolo de inspección preventiva.")
    else:
        recommendations.append("Mantener monitoreo operativo normal.")

    if probability >= 0.85:
        recommendations.append("Riesgo crítico: revisar máquina de forma inmediata.")
    elif probability >= 0.70:
        recommendations.append("Riesgo alto: programar mantenimiento en las próximas 24 horas.")
    elif probability >= 0.50:
        recommendations.append("Riesgo medio: aumentar frecuencia de monitoreo.")
    else:
        recommendations.append("Riesgo bajo: operación dentro de parámetros aceptables.")

    if torque > 55:
        recommendations.append("Torque elevado: revisar carga mecánica y fricción.")
    if rpm < 1200:
        recommendations.append("RPM bajas: revisar motor, transmisión o resistencia mecánica.")
    if tool_wear > 180:
        recommendations.append("Desgaste alto de herramienta: considerar reemplazo.")
    if process_temp - air_temp > 12:
        recommendations.append("Diferencia térmica elevada: revisar sistema de enfriamiento.")

    return recommendations


def explain_with_shap(input_df):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    input_transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_transformed)

    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]
    else:
        shap_values_class = shap_values

    shap_array = np.array(shap_values_class)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    shap_row = shap_array[0].ravel()

    shap_df = pd.DataFrame({
        "Variable": feature_names,
        "Impacto_SHAP": shap_row
    })

    shap_df["Impacto_abs"] = shap_df["Impacto_SHAP"].abs()
    shap_df = shap_df.sort_values("Impacto_abs", ascending=False)

    return shap_df


init_db()


st.title("⚙️ Plataforma de Mantenimiento Predictivo")
st.write("Predicción de falla, explicación SHAP, recomendaciones operativas e historial de resultados.")

tab1, tab2, tab3 = st.tabs([
    "Predicción individual",
    "Predicción batch",
    "Dashboard operacional"
])


with tab1:
    st.sidebar.header("Parámetros de la máquina")

    machine_type = st.sidebar.selectbox("Tipo de máquina", ["L", "M", "H"])
    air_temperature = st.sidebar.number_input("Air temperature [K]", 250.0, 350.0, 298.1)
    process_temperature = st.sidebar.number_input("Process temperature [K]", 250.0, 400.0, 308.6)
    rotational_speed = st.sidebar.number_input("Rotational speed [rpm]", 0, 4000, 1551)
    torque = st.sidebar.number_input("Torque [Nm]", 0.0, 100.0, 42.8)
    tool_wear = st.sidebar.number_input("Tool wear [min]", 0, 300, 0)

    payload = {
        "Type": machine_type,
        "Air_temperature_K": air_temperature,
        "Process_temperature_K": process_temperature,
        "Rotational_speed_rpm": rotational_speed,
        "Torque_Nm": torque,
        "Tool_wear_min": tool_wear
    }

    input_df = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Datos enviados a la API")
        st.json(payload)

    with col_b:
        st.subheader("Datos usados por el modelo")
        st.dataframe(input_df, use_container_width=True)

    if st.button("Predecir y explicar"):
        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()

                probability = result["probability"]
                failure_expected = result["failure_expected"]
                risk_level = get_risk_level(probability)

                save_prediction(payload, result, risk_level)

                st.divider()
                st.subheader("Resultado")

                c1, c2, c3 = st.columns(3)

                with c1:
                    if failure_expected:
                        st.error("⚠️ Riesgo de falla detectado")
                    else:
                        st.success("✅ No se espera falla")

                with c2:
                    st.metric("Probabilidad", f"{probability * 100:.2f}%")

                with c3:
                    st.metric("Nivel de riesgo", risk_level)

                st.plotly_chart(create_gauge(probability, risk_level), use_container_width=True)

                st.subheader("Recomendaciones operativas")
                recommendations = get_recommendations(input_df, probability, failure_expected)

                for rec in recommendations:
                    st.write(f"• {rec}")

                st.divider()
                st.subheader("Explicación SHAP")

                shap_df = explain_with_shap(input_df)

                st.dataframe(
                    shap_df[["Variable", "Impacto_SHAP"]].head(10),
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                shap_df.head(10).sort_values("Impacto_SHAP").plot(
                    kind="barh",
                    x="Variable",
                    y="Impacto_SHAP",
                    ax=ax,
                    legend=False
                )
                ax.set_title("Impacto de variables sobre la predicción")
                ax.set_xlabel("Valor SHAP")
                ax.set_ylabel("Variable")
                st.pyplot(fig)

                st.info(
                    "Valores SHAP positivos empujan la predicción hacia falla. "
                    "Valores negativos empujan la predicción hacia no falla."
                )

            else:
                st.error("Error en la respuesta de la API")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con FastAPI. Ejecuta primero: uvicorn app.main:app --reload")


with tab2:
    st.subheader("Predicción batch desde CSV")

    st.write("El archivo debe contener columnas compatibles con el modelo:")
    st.code("""
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
""")

    uploaded_file = st.file_uploader("Carga un archivo CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Vista previa:")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Ejecutar predicción batch"):
            results = []

            for _, row in batch_df.iterrows():
                payload_batch = {
                    "Type": row["Type"],
                    "Air_temperature_K": float(row["Air temperature [K]"]),
                    "Process_temperature_K": float(row["Process temperature [K]"]),
                    "Rotational_speed_rpm": int(row["Rotational speed [rpm]"]),
                    "Torque_Nm": float(row["Torque [Nm]"]),
                    "Tool_wear_min": int(row["Tool wear [min]"])
                }

                response = requests.post(API_URL, json=payload_batch)

                if response.status_code == 200:
                    result = response.json()
                    risk_level = get_risk_level(result["probability"])

                    results.append({
                        **payload_batch,
                        "prediction": result["prediction"],
                        "failure_expected": result["failure_expected"],
                        "probability": result["probability"],
                        "risk_level": risk_level
                    })

            results_df = pd.DataFrame(results)

            st.subheader("Resultados batch")
            st.dataframe(results_df, use_container_width=True)

            csv = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Descargar resultados",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )


with tab3:
    st.subheader("Dashboard operacional")

    history_df = load_history()

    if history_df.empty:
        st.info("Todavía no hay predicciones guardadas.")
    else:
        total_predictions = len(history_df)
        total_failures = history_df["failure_expected"].sum()
        avg_probability = history_df["probability"].mean()
        critical_cases = (history_df["risk_level"] == "Crítico").sum()

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Total predicciones", total_predictions)
        k2.metric("Casos con riesgo", int(total_failures))
        k3.metric("Probabilidad promedio", f"{avg_probability * 100:.2f}%")
        k4.metric("Casos críticos", int(critical_cases))

        st.divider()

        st.subheader("Historial de predicciones")
        st.dataframe(history_df, use_container_width=True)

        st.subheader("Distribución de niveles de riesgo")
        risk_counts = history_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        fig_risk = go.Figure(
            data=[
                go.Bar(
                    x=risk_counts["risk_level"],
                    y=risk_counts["count"]
                )
            ]
        )
        fig_risk.update_layout(
            title="Predicciones por nivel de riesgo",
            xaxis_title="Nivel de riesgo",
            yaxis_title="Cantidad"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        st.subheader("Evolución de probabilidad")
        history_df["created_at"] = pd.to_datetime(history_df["created_at"])

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=history_df["created_at"],
            y=history_df["probability"],
            mode="lines+markers",
            name="Probabilidad"
        ))
        fig_line.update_layout(
            title="Probabilidad de riesgo en el tiempo",
            xaxis_title="Fecha",
            yaxis_title="Probabilidad"
        )
        st.plotly_chart(fig_line, use_container_width=True)