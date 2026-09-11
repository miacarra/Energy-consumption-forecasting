# Energy Consumption Forecasting with LSTM

This is a small course project exploring energy-consumption forecasting with a Long
Short-Term Memory (LSTM) neural network. The model uses temperature and a
synthetic economic index to predict the next energy-consumption value from the
previous 24 observations.

## Project status

The repository preserves the original course experiment and adds a cleaned,
reproducible implementation. The included dataset is hypothetical and contains
1,095 daily observations from 1 January 2020 to 30 December 2022.

## Repository structure

```text
assets/                         Original course-project chart
data/hypothetical_energy_data.csv
legacy/new_lstm_original.py     Original course implementation
src/train_lstm.py               Cleaned training and evaluation script
requirements.txt
```

## Improvements in the cleaned version

- keeps the chronological order when creating the train/test split;
- fits scalers only on the training period to prevent data leakage;
- scales the target separately and converts predictions correctly;
- uses fixed random seeds and disables shuffling during training;
- reports RMSE and MAE;
- saves the trained model and generated charts to a local ignored directory.

## Run locally

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python src/train_lstm.py
```

Training creates a model and two charts inside `outputs/`. That directory is
ignored by Git because these files are generated artifacts.

## Notes

The dataset is synthetic, so the results demonstrate the modeling workflow and
should not be interpreted as forecasts of real-world energy demand.

