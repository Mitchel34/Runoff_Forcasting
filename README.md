# Runoff Forecasting: NWM Error Correction with Deep Learning

This project implements deep learning models to improve National Water Model (NWM) runoff forecasts through error prediction. It uses specialized architectures for two distinct USGS stations:
- **Station 21609641**: LSTM-based error prediction model
- **Station 20380357**: Transformer-based error prediction model

## Project Structure

```
runoff_forecasting/
├── data/
│   ├── raw/                  # Original NWM and USGS data
│   └── processed/            # Preprocessed data ready for training
├── src/
│   ├── preprocess.py         # Data preprocessing pipeline
│   ├── models/
│   │   ├── lstm.py           # LSTM implementation for station 21609641
│   │   └── transformer.py    # Transformer implementation for station 20380357
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Evaluation and visualization
│   └── utils.py              # Utility functions and metrics
├── notebooks/                # Exploratory data analysis
├── results/                  # Saved results and visualizations
│   ├── plots/                # Generated visualizations
│   └── metrics/              # Performance metrics
├── models/                   # Saved model files
├── report/                   # Technical report
└── requirements.txt          # Project dependencies
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd runoff_forecasting
    ```

2.  **Create and activate a Python virtual environment:**
    It is recommended to use Python 3.10 for compatibility with the specified dependencies (especially `tensorflow-macos`).
    ```bash
    python3.10 -m venv .venv
    source .venv/bin/activate 
    # On Windows use: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

The project requires NWM forecast data and USGS observation data for two stations:
- Station 21609641 (USGS gauge 11266500)
- Station 20380357 (USGS gauge 09520500)

## Data Preprocessing

Raw NWM forecast data and USGS observation data should be placed in the `data/raw/<station_id>/` directories. The expected structure is shown in the Project Structure section.

To preprocess the data for both stations (20380357 and 21609641), run the preprocessing script:

```bash
python src/preprocess.py
```

This script will:
- Load raw NWM and USGS data.
- Align timestamps and merge the datasets.
- Calculate forecast error (`usgs_flow - nwm_flow`).
- Pivot the data to have lead times (1-18h) as columns.
- Create input sequences (24hr lookback) and corresponding target sequences (18hr forecast error).
- Split data into training (Apr 2021 - Sep 2022) and testing (Oct 2022 - Apr 2023) sets.
- Scale features and targets using `StandardScaler` (fitted on training data only).
- Save the processed data (`.npz` files) and scalers (`.joblib` files) to the `data/processed/` directory.

Default arguments:
- `--data-dir`: `data/raw`
- `--output-dir`: `data/processed`
- `--window-size`: `24` (hours)
- `--horizon`: `18` (hours)

You can override these defaults if needed, e.g.:
```bash
python src/preprocess.py --window-size 48 --output-dir data/processed_ws48
```

## Workflow

1. **Preprocess data**:
```bash
python src/preprocess.py
```

2. **Train models**:
```bash
python src/train.py --station 21609641  # LSTM for station 21609641
python src/train.py --station 20380357  # Transformer for station 20380357
```

3. **Evaluate and visualize results**:
```bash
python src/evaluate.py
```

## Evaluation Metrics

The models are evaluated using the following hydrological metrics:
- Correlation Coefficient (CC)
- Root Mean Square Error (RMSE)
- Percent Bias (PBIAS)
- Nash-Sutcliffe Efficiency (NSE)

## Authors

Mitchel Carson & Christian Castaneda

## License

[MIT](LICENSE)

## Acknowledgments

This project is based on methods from:
- Han, H. & Morrison, R. R. (2022). Improved runoff forecasting performance through error predictions using a deep-learning approach. Journal of Hydrology, 608, 127653.
