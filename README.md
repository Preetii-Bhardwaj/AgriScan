# AgriScan

Commodity price forecasting and dispatch net-profit optimization system built with **Python**, **TensorFlow/Keras (Bi-LSTM)**, **MinMaxScaler**, **FastAPI**, and **Pydantic**.

## Overview
- **Data Preparation:** Daily calendar alignment with linear interpolation, chronological split, leak-free `MinMaxScaler`, and 30-day sliding windows.
- **Forecasting:** 64-unit `Bidirectional(LSTM)` model evaluated using `MAE` and `RMSE` in `₹/qtl`.
- **Profit Optimization:** Evaluates `net_profit = gross_revenue - (transport + fixed_trip + holding_costs)` across wait horizons (`0`, `1`, `3`, `7` days).
- **API:** FastAPI service (`phase4_api.py`) serving `/health`, `/metrics`, `/forecast`, and `/optimize-dispatch`.
