from contextlib import asynccontextmanager
import json
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

ARTIFACTS_DIR = "artifacts"
CLEANED_SERIES_PATH = os.path.join(ARTIFACTS_DIR, "cleaned_series.csv")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "price_scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "bilstm_model.keras")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
PROFIT_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "profit_decision.json")

LOOKBACK_DAYS = 30
HORIZONS = [1, 3, 7]

DEFAULT_QUANTITY_QUINTALS = 10.0
DEFAULT_TRANSPORT_COST_PER_QUINTAL = 120.0
DEFAULT_FIXED_TRIP_COST = 1500.0
DEFAULT_DAILY_HOLDING_COST_PER_QUINTAL = 15.0


class OptimizeRequest(BaseModel):
    quantity_quintals: float = Field(default=DEFAULT_QUANTITY_QUINTALS, gt=0)
    transport_cost_per_quintal: float = Field(default=DEFAULT_TRANSPORT_COST_PER_QUINTAL, ge=0)
    fixed_trip_cost: float = Field(default=DEFAULT_FIXED_TRIP_COST, ge=0)
    daily_holding_cost_per_quintal: float = Field(default=DEFAULT_DAILY_HOLDING_COST_PER_QUINTAL, ge=0)


def forecast_horizons(series_df, scaler, model, lookback_days=LOOKBACK_DAYS, horizons=HORIZONS):
    max_horizon = max(horizons)
    last_window_df = series_df[["Modal_Price"]].iloc[-lookback_days:].copy()
    rolling_scaled = scaler.transform(last_window_df).reshape(1, lookback_days, 1)

    forecasts_by_step = []
    for _ in range(max_horizon):
        next_scaled = float(model.predict(rolling_scaled, verbose=0)[0, 0])
        next_price_df = pd.DataFrame([[next_scaled]], columns=["Modal_Price"])
        next_price_rupees = float(scaler.inverse_transform(next_price_df)[0, 0])
        forecasts_by_step.append(round(next_price_rupees, 2))

        next_step_tensor = np.array([[[next_scaled]]], dtype=float)
        rolling_scaled = np.concatenate([rolling_scaled[:, 1:, :], next_step_tensor], axis=1)

    return {h: forecasts_by_step[h - 1] for h in horizons}


def compute_net_profit(
    price_per_quintal: float,
    wait_days: int,
    quantity_quintals: float,
    transport_cost_per_quintal: float,
    fixed_trip_cost: float,
    daily_holding_cost_per_quintal: float,
) -> dict:
    gross_revenue = price_per_quintal * quantity_quintals
    variable_transport = transport_cost_per_quintal * quantity_quintals
    holding_cost = daily_holding_cost_per_quintal * quantity_quintals * wait_days
    total_logistics_cost = variable_transport + fixed_trip_cost + holding_cost
    net_profit = gross_revenue - total_logistics_cost

    return {
        "wait_days": int(wait_days),
        "predicted_modal_price_per_quintal": round(float(price_per_quintal), 2),
        "gross_revenue_rupees": round(float(gross_revenue), 2),
        "total_logistics_cost_rupees": round(float(total_logistics_cost), 2),
        "net_profit_rupees": round(float(net_profit), 2),
    }


def build_decision_payload(state: dict, req: OptimizeRequest) -> dict:
    series_df = state["series_df"]
    scaler = state["scaler"]
    model = state["model"]

    latest_date = series_df["Price Date"].iloc[-1].strftime("%Y-%m-%d")
    current_price = float(series_df["Modal_Price"].iloc[-1])

    horizon_prices = forecast_horizons(series_df, scaler, model, LOOKBACK_DAYS, HORIZONS)

    candidates = [
        compute_net_profit(
            current_price,
            wait_days=0,
            quantity_quintals=req.quantity_quintals,
            transport_cost_per_quintal=req.transport_cost_per_quintal,
            fixed_trip_cost=req.fixed_trip_cost,
            daily_holding_cost_per_quintal=req.daily_holding_cost_per_quintal,
        )
    ]

    for h in HORIZONS:
        candidates.append(
            compute_net_profit(
                horizon_prices[h],
                wait_days=h,
                quantity_quintals=req.quantity_quintals,
                transport_cost_per_quintal=req.transport_cost_per_quintal,
                fixed_trip_cost=req.fixed_trip_cost,
                daily_holding_cost_per_quintal=req.daily_holding_cost_per_quintal,
            )
        )

    best_option = max(candidates, key=lambda item: item["net_profit_rupees"])

    return {
        "as_of_date": latest_date,
        "current_modal_price_per_quintal": round(current_price, 2),
        "quantity_quintals": req.quantity_quintals,
        "cost_assumptions_rupees": {
            "transport_cost_per_quintal": req.transport_cost_per_quintal,
            "fixed_trip_cost": req.fixed_trip_cost,
            "daily_holding_cost_per_quintal": req.daily_holding_cost_per_quintal,
        },
        "forecast_prices_rupees": horizon_prices,
        "evaluated_options": candidates,
        "recommended_action": best_option,
    }


app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (
        os.path.exists(CLEANED_SERIES_PATH)
        and os.path.exists(SCALER_PATH)
        and os.path.exists(MODEL_PATH)
    ):
        raise RuntimeError("Missing required Phase 1/2 artifacts in artifacts/ directory.")

    series_df = pd.read_csv(CLEANED_SERIES_PATH)
    series_df["Price Date"] = pd.to_datetime(series_df["Price Date"])
    series_df = series_df.sort_values("Price Date").dropna(subset=["Modal_Price"])

    if len(series_df) < LOOKBACK_DAYS:
        raise RuntimeError(f"Need at least {LOOKBACK_DAYS} rows in cleaned_series.csv.")

    app_state["series_df"] = series_df
    app_state["scaler"] = joblib.load(SCALER_PATH)
    app_state["model"] = load_model(MODEL_PATH)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            app_state["metrics"] = json.load(f)
    else:
        app_state["metrics"] = {}

    yield
    app_state.clear()


app = FastAPI(title="AgriScan API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": "model" in app_state,
        "records_available": int(len(app_state["series_df"])) if "series_df" in app_state else 0,
    }


@app.get("/metrics")
def get_model_metrics():
    if not app_state.get("metrics"):
        raise HTTPException(status_code=404, detail="metrics.json not found.")
    return app_state["metrics"]


@app.get("/forecast")
def get_forecast():
    try:
        prices = forecast_horizons(
            series_df=app_state["series_df"],
            scaler=app_state["scaler"],
            model=app_state["model"],
        )
        latest_date = app_state["series_df"]["Price Date"].iloc[-1].strftime("%Y-%m-%d")
        current_price = round(float(app_state["series_df"]["Modal_Price"].iloc[-1]), 2)
        return {
            "as_of_date": latest_date,
            "current_modal_price_per_quintal": current_price,
            "forecast_prices_rupees": prices,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/optimize-dispatch")
def optimize_dispatch_endpoint(req: OptimizeRequest):
    try:
        payload = build_decision_payload(app_state, req)
        with open(PROFIT_OUTPUT_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
