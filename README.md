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
│   ├── tune.py               # Hyperparameter tuning script
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

## Hyperparameter Tuning

Before final training, hyperparameter tuning is performed using `src/tune.py` to find the optimal configuration for each model (LSTM and Transformer).

-   **Library:** Keras Tuner
-   **Strategy:** A hybrid approach is recommended:
    1.  **Hyperband:** Run first for broad exploration (`--tuner hyperband`).
    2.  **Bayesian Optimization:** Run second to refine promising configurations (`--tuner bayesian`).
-   **Process:** The script trains multiple model versions with different hyperparameter combinations (learning rate, layer sizes, dropout, etc.) on a validation split of the training data, aiming to minimize validation loss.
-   **Output:** The best hyperparameters found are printed to the console and saved to a JSON file in the `results/hyperparameters/` directory (e.g., `results/hyperparameters/21609641_lstm_best_hps.json`). This file can be used to configure the final training in `train.py`.

**Example Tuning Commands (run from project root):**

```bash
# 1. Hyperband Tuning (Example for LSTM)
python src/tune.py --station_id 21609641 --model_type lstm --tuner hyperband --epochs_per_trial 50 --batch_size 64

# 2. Bayesian Optimization Tuning (Example for Transformer, refining after Hyperband)
python src/tune.py --station_id 20380357 --model_type transformer --tuner bayesian --max_trials 30 --epochs_per_trial 80 --batch_size 64
```

## Workflow

1.  **Preprocess data**:
    ```bash
    python src/preprocess.py
    ```

2.  **(Optional but Recommended) Hyperparameter Tuning**:
    *   Run Hyperband:
        ```bash
        python src/tune.py --station_id <station_id> --model_type <lstm|transformer> --tuner hyperband ...
        ```
    *   Run Bayesian Optimization (potentially refining search space based on Hyperband):
        ```bash
        python src/tune.py --station_id <station_id> --model_type <lstm|transformer> --tuner bayesian ...
        ```
    *   Note the best hyperparameters found for each station/model.

3.  **Train final models** (using best hyperparameters from tuning):
    ```bash
    # Example for LSTM (Station 21609641)
    python src/train.py --station_id 21609641 --model_type lstm --epochs 100 --batch_size <best_batch> --lr <best_lr> --lstm_units <best_units> # Add other tuned params

    # Example for Transformer (Station 20380357)
    python src/train.py --station_id 20380357 --model_type transformer --epochs 100 --batch_size <best_batch> --lr <best_lr> --tf_heads <best_heads> --tf_ff_dim <best_ff_dim> --tf_blocks <best_blocks> --tf_dropout <best_dropout> # Add other tuned params
    ```

4.  **Evaluate and visualize results**:
    ```bash
    # Evaluate LSTM
    python src/evaluate.py --station_id 21609641 --model_type lstm

    # Evaluate Transformer
    python src/evaluate.py --station_id 20380357 --model_type transformer
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
