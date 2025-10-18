from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import Flask, jsonify, render_template, request
from modelo_forecast import forecast_center_date

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_ingresos.pkl"
DEFAULT_DATA_CANDIDATES = [
    BASE_DIR / "ingresos.csv",
    BASE_DIR / "datos_anonimizados.csv",
]


def _load_history() -> pd.DataFrame:
    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return pd.read_csv(candidate, parse_dates=["timestamp"])
    raise FileNotFoundError(
        "No se encontró un archivo de datos. "
        "Coloca 'ingresos.csv' (o 'datos_anonimizados.csv') junto a app.py."
    )


def to_float32(matrix):
    """Helper usado en el pipeline entrenado para castear a float32."""
    return np.asarray(matrix, dtype=np.float32)


# Garantizar que joblib pueda resolver la referencia pickled
import sys
sys.modules[__name__].to_float32 = to_float32


app = Flask(__name__)
model = joblib.load(MODEL_PATH)
data = _load_history()


def _centros_disponibles() -> list[int]:
    return (
        pd.Series(data["id_cc"].dropna().unique(), dtype="Int64")
        .dropna()
        .astype(int)
        .sort_values()
        .tolist()
    )


@app.route("/")
def home():
    return render_template(
        "index.html",
        centros=_centros_disponibles(),
        selected_table="puertas",
        table_payload=json.dumps({}, ensure_ascii=False),
        plot_detalle="",
        plot_total="",
        plot_daily="",
    )


@app.route("/predict", methods=["POST"])
def predict():
    centros = _centros_disponibles()
    table_view = request.form.get("table_view", "puertas")

    try:
        id_cc = int(request.form["id_cc"])
        fecha_inicio = request.form["fecha_inicio"]
        fecha_fin = request.form.get("fecha_fin") or None
        datetime.strptime(fecha_inicio, "%Y-%m-%d")
        if fecha_fin:
            datetime.strptime(fecha_fin, "%Y-%m-%d")
    except (ValueError, KeyError):
        return (
            render_template(
                "index.html",
                centros=centros,
                error="Debes proporcionar un id_cc válido y fechas en formato YYYY-MM-DD.",
                selected_cc=request.form.get("id_cc"),
                selected_start=request.form.get("fecha_inicio"),
                selected_end=request.form.get("fecha_fin"),
                selected_table=table_view,
            ),
            400,
        )

    try:
        forecast = forecast_center_date(
            data[["timestamp", "acceso_id", "id_cc", "ins"]],
            model,
            id_cc,
            fecha_inicio,
            end_date=fecha_fin,
        )
    except ValueError as exc:
        return (
            render_template(
                "index.html",
                centros=centros,
                error=str(exc),
                selected_cc=id_cc,
                selected_start=fecha_inicio,
                selected_end=fecha_fin,
                selected_table=table_view,
            ),
            404,
        )
    except Exception as exc:  # pragma: no cover
        return (
            render_template(
                "index.html",
                centros=centros,
                error=f"Error al generar la predicción: {exc}",
                selected_cc=id_cc,
                selected_start=fecha_inicio,
                selected_end=fecha_fin,
                selected_table=table_view,
            ),
            500,
        )

    rango_texto = (
        f"{fecha_inicio} → {fecha_fin}" if fecha_fin else fecha_inicio
    )

    forecast["Ingresos"] = forecast["yhat"].round().astype(int)
    forecast = forecast.drop(columns=["yhat"])

    fig_detalle = px.line(
        forecast,
        x="timestamp",
        y="Ingresos",
        color="acceso_id",
        title=f"Predicción por acceso - Centro {id_cc} ({rango_texto})",
        labels={"timestamp": "Hora", "Ingresos": "Ingresos estimados", "acceso_id": "Acceso"},
    )
    fig_detalle.update_layout(legend_title="Acceso", template="plotly_white")
    fig_detalle.update_yaxes(tickformat=",.0f")
    fig_detalle.update_traces(hovertemplate="Hora=%{x}<br>Ingresos=%{y:,}<extra></extra>")
    plot_detalle = pio.to_html(fig_detalle, full_html=False)

    total_forecast = (
        forecast.groupby("timestamp", as_index=False)["Ingresos"]
        .sum()
        .rename(columns={"Ingresos": "Ingresos"})
    )
    total_forecast["Ingresos"] = total_forecast["Ingresos"].round().astype(int)

    fig_total = px.line(
        total_forecast,
        x="timestamp",
        y="Ingresos",
        title=f"Predicción total - Centro {id_cc} ({rango_texto})",
        labels={"timestamp": "Hora", "Ingresos": "Ingresos estimados totales"},
    )
    fig_total.update_layout(template="plotly_white")
    fig_total.update_yaxes(tickformat=",.0f")
    fig_total.update_traces(hovertemplate="Hora=%{x}<br>Total=%{y:,}<extra></extra>")
    plot_total = pio.to_html(fig_total, full_html=False)

    daily_total = (
        total_forecast.assign(fecha=total_forecast["timestamp"].dt.date)
        .groupby("fecha", as_index=False)["Ingresos"]
        .sum()
        .rename(columns={"Ingresos": "Ingresos"})
    )
    daily_total["Ingresos"] = daily_total["Ingresos"].round().astype(int)

    fig_daily = px.line(
        daily_total,
        x="fecha",
        y="Ingresos",
        title=f"Total diario - Centro {id_cc} ({rango_texto})",
        labels={"fecha": "Fecha", "Ingresos": "Ingresos estimados diarios"},
    )
    fig_daily.update_traces(mode="lines+markers")
    fig_daily.update_layout(template="plotly_white")
    fig_daily.update_yaxes(tickformat=",.0f")
    fig_daily.update_traces(hovertemplate="Fecha=%{x}<br>Total diario=%{y:,}<extra></extra>")
    plot_daily = pio.to_html(fig_daily, full_html=False)

    def df_payload(df: pd.DataFrame) -> dict[str, object]:
        serial = df.copy()
        if "timestamp" in serial.columns:
            serial["timestamp"] = serial["timestamp"].astype(str)
        if "fecha" in serial.columns:
            serial["fecha"] = serial["fecha"].astype(str)
        numeric_cols = serial.select_dtypes(include=["float", "int"]).columns
        for col in numeric_cols:
            serial[col] = serial[col].round().astype(int)
        return {
            "columns": serial.columns.tolist(),
            "data": serial.to_dict(orient="records"),
        }

    table_payload = {
        "puertas": df_payload(forecast),
        "horaria": df_payload(total_forecast),
        "diaria": df_payload(daily_total),
    }

    return render_template(
        "index.html",
        centros=centros,
        selected_cc=id_cc,
        selected_start=fecha_inicio,
        selected_end=fecha_fin,
        selected_table=table_view,
        plot_detalle=plot_detalle,
        plot_total=plot_total,
        plot_daily=plot_daily,
        table_payload=json.dumps(table_payload, ensure_ascii=False),
    )


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(debug=True)
