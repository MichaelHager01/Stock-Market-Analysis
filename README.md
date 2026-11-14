# Stock market analysis

Forecasts Microsoft (MSFT) closing prices from the previous three trading days with an LSTM.

## What it does

- Builds date-ordered windows with `df_to_windowed_df` and `n=3`.
- Trains a Keras `LSTM(64)` on mean squared error, with a held-out validation slice and a later test slice.
- Fits for 100 epochs, then continues into chained multi-day forecasts.
- The Yahoo Finance export is not committed. Save it as `data/MSFT (1).csv`, or leave it next to the notebook if you run this in Colab.

## Layout

- `notebooks/analysis.ipynb` — the windowing, the LSTM, and the forecasts

## Run

From the repository root, after the CSV is in place:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

## Stack

Python, pandas, NumPy, TensorFlow, Matplotlib, Jupyter. Team project with Santiago.
