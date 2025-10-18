"""
Herramientas de pronóstico horario por centro comercial.

Este módulo encapsula la lógica utilizada en el notebook `model_ingresos_por_cc.ipynb`
para poder reutilizarla desde aplicaciones (por ejemplo, la app Flask).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "hour",
    "dow",
    "dom",
    "week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "roll_mean_24",
    "roll_std_24",
    "roll_mean_168",
    "roll_std_168",
    "acceso_id",
    "id_cc",
]

LAG_COLUMNS = ["lag_1", "lag_24", "lag_168"]
ROLL_COLUMNS = ["roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168"]


@dataclass(slots=True)
class _HistoryConfig:
    """Parámetros para el procesamiento de series por acceso."""

    freq: str = "h"
    fill_value: float = 0.0


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas y tipos básicos."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["acceso_id"] = df["acceso_id"].astype("string")
    df["id_cc"] = df["id_cc"].astype("Int64")
    df = df.sort_values(["acceso_id", "timestamp"]).reset_index(drop=True)
    return df


def _ensure_hourly_history(df: pd.DataFrame, config: _HistoryConfig) -> pd.DataFrame:
    """Garantiza una frecuencia horaria continua por acceso."""
    pieces = []
    for acceso, grp in df.groupby("acceso_id", sort=False):
        g = (
            grp.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .set_index("timestamp")
            .sort_index()
        )
        g = g[["acceso_id", "id_cc", "ins"]].asfreq(config.freq)
        g["acceso_id"] = g["acceso_id"].ffill().bfill().astype("string")
        g["id_cc"] = g["id_cc"].ffill().bfill().astype("Int64")
        g["ins"] = (
            g["ins"]
            .ffill()
            .bfill()
            .astype(float)
            .fillna(config.fill_value)
        )
        pieces.append(g.reset_index())
    history = pd.concat(pieces, ignore_index=True)
    history = history.sort_values(["acceso_id", "timestamp"]).reset_index(drop=True)
    return history


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["dow"] = out["timestamp"].dt.dayofweek
    out["dom"] = out["timestamp"].dt.day
    out["week"] = out["timestamp"].dt.isocalendar().week.astype(int)
    out["month"] = out["timestamp"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def _add_lag_roll_features(
    df: pd.DataFrame,
    lags: Iterable[int] = (1, 24, 168),
    roll_windows: Iterable[int] = (24, 168),
) -> pd.DataFrame:
    out = df.sort_values(["acceso_id", "timestamp"]).copy()
    for lag in lags:
        out[f"lag_{lag}"] = out.groupby("acceso_id")["ins"].shift(lag)
    for window in roll_windows:
        shifted = out.groupby("acceso_id")["ins"].shift(1)
        rolled = shifted.groupby(out["acceso_id"]).rolling(
            window=window, min_periods=max(2, window // 4)
        )
        out[f"roll_mean_{window}"] = rolled.mean().reset_index(level=0, drop=True)
        out[f"roll_std_{window}"] = rolled.std().reset_index(level=0, drop=True)
    return out


def _fill_feature_nans(row: pd.Series, history: pd.DataFrame) -> pd.Series:
    """Rellena NaN en lags/rolling con valores razonables."""
    for col in LAG_COLUMNS:
        if pd.isna(row[col]):
            fallback = history["ins"].iloc[-24:].mean() if len(history) else 0.0
            row[col] = float(np.nan_to_num(fallback))
    for col in ROLL_COLUMNS:
        if pd.isna(row[col]):
            if "std" in col:
                fallback = history["ins"].iloc[-168:].std(ddof=0) if len(history) else 0.0
            else:
                fallback = history["ins"].iloc[-24:].mean() if len(history) else 0.0
            row[col] = float(np.nan_to_num(fallback))
    return row


def forecast_center_date(
    df_hist: pd.DataFrame,
   model_pipe: Pipeline,
   center_id: int,
   target_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Pronostica horas por acceso de un centro para una fecha o rango.

    Parameters
    ----------
    df_hist:
        Historial con las columnas `timestamp`, `acceso_id`, `id_cc`, `ins`.
    model_pipe:
        Pipeline de sklearn entrenado (incluye la transformación de categorías).
    center_id:
        Identificador del centro comercial (`id_cc`).
    target_date:
        Fecha inicial (string o datetime). Se pronostican 24 horas desde las 00:00
        si no se entrega `end_date`.
    end_date:
        Fecha final opcional (inclusive). Si se especifica, se generan predicciones
        horarias continuas hasta las 23:00 del día final.

    Returns
    -------
    DataFrame con columnas `id_cc`, `acceso_id`, `timestamp`, `yhat`.
    """

    history = _ensure_datetime(df_hist)
    history = history[history["id_cc"] == center_id]
    if history.empty:
        raise ValueError(f"No hay accesos disponibles para id_cc={center_id}.")

    history = _ensure_hourly_history(history, _HistoryConfig())

    start_ts = pd.to_datetime(target_date).normalize()
    if end_date is None:
        end_ts = start_ts
    else:
        end_ts = pd.to_datetime(end_date).normalize()
        if end_ts < start_ts:
            raise ValueError("La fecha final debe ser mayor o igual a la inicial.")

    horizon = pd.date_range(start_ts, end_ts + pd.Timedelta(hours=23), freq="h")

    predictions = []
    for acceso_id, grp in history.groupby("acceso_id", sort=True):
        cur = grp.set_index("timestamp").copy()
        cur = cur.sort_index()

        for ts in horizon:
            # asegurar que el timestamp actual existe para cálculo de features
            if ts not in cur.index:
                cur.loc[ts, ["acceso_id", "id_cc"]] = (acceso_id, center_id)
            cur = cur.sort_index()

            row = cur.loc[[ts]].reset_index().rename(columns={"index": "timestamp"})
            row = _add_calendar_features(row)
            augmented = cur.reset_index().rename(columns={"index": "timestamp"})
            augmented = _add_calendar_features(augmented)
            augmented = _add_lag_roll_features(augmented)
            row = augmented[augmented["timestamp"] == ts].copy()

            row = row.apply(
                lambda s: _fill_feature_nans(s, cur.reset_index()), axis=1
            )

            missing = [c for c in FEATURE_COLUMNS if c not in row.columns]
            if missing:
                raise KeyError(
                    f"Faltan columnas para el modelo: {', '.join(missing)}"
                )

            yhat = float(model_pipe.predict(row[FEATURE_COLUMNS])[0])
            yhat = max(0.0, yhat)

            predictions.append(
                {
                    "id_cc": center_id,
                    "acceso_id": acceso_id,
                    "timestamp": ts,
                    "yhat": yhat,
                }
            )

            cur.loc[ts, "ins"] = yhat

    result = pd.DataFrame(predictions).sort_values(
        ["acceso_id", "timestamp"]
    ).reset_index(drop=True)

    return result


__all__ = ["forecast_center_date", "FEATURE_COLUMNS"]
