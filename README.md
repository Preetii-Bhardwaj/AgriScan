# AgriScan — Commodity Price Forecasting & Dispatch Profit Optimizer

End-to-end machine learning and decision-support system that forecasts agricultural mandi prices and recommends optimal selling horizons (`0`, `1`, `3`, or `7` days) by factoring in logistics and holding costs.

## Architecture
1. **Data Preparation:** Daily time-series alignment with linear interpolation, chronological train/test split, leak-free `MinMaxScaler`, and 30-day sliding window sequences.
2. **Forecasting Engine:** 64-unit `Bidirectional(LSTM)` model trained on historical mandi price sequences (`MAE: ₹269.24`, `RMSE: ₹333.48`).
3. **Profit Optimization:** Evaluates expected `net_profit = gross_revenue - (transport + fixed_trip + holding_costs)` across wait horizons to recommend the highest-margin dispatch day.
4. **API Service:** FastAPI backend (`phase4_api.py`) with Pydantic request/response validation.

## API Endpoints
- `GET /health` — Service health check
- `GET /metrics` — Model evaluation metrics (`MAE`, `RMSE`)
- `POST /forecast` — Multi-horizon commodity price forecast
- `POST /optimize-dispatch` — Net-profit comparison and optimal selling window recommendation

## Quickstart
```bash
pip install -r requirements.txt
uvicorn phase4_api:app --reload
