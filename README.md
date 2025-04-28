<<<<<<< HEAD
# Improving NWM Forecasts Using Deep Learning

## Project Overview

This project enhances the accuracy of the National Water Model (NWM) short-range runoff forecasts through deep learning-based post-processing. By training a model to predict errors in NWM forecasts, we correct these forecasts to align closely with observed runoff data from the United States Geological Survey (USGS). The project focuses on two US stations, using data from April 2021 to April 2023, and targets forecast lead times of 1 to 18 hours.

## Objectives

1.  **Preprocess Data:** Clean and align NWM forecasts and USGS observations, splitting into training, validation, and test sets.
2.  **Develop Model:** Implement a deep learning model, such as Long Short-Term Memory (LSTM), to predict forecast errors.
3.  **Evaluate Performance:** Compare corrected forecasts against raw NWM forecasts and observed data using hydrologic metrics.
4.  **Visualize Results:** Generate box-plots for runoff values and plots for evaluation metrics across lead times.

## Data Sources


*   **NWM Forecasts:** Hourly short-range forecasts (lead times 1–18 hours) for two US stations, from April 2021 to April 2023. Files are named `streamflow_[streamID]_[YYYYMM].csv` and include columns like `model_output_valid_time` and `streamflow_value`.
*   **USGS Observations:** Hourly observed runoff data for the same period, in CSV files named `*_Strt_*.csv`, with columns like `DateTime` and `USGSFlowValue`.
*   **Optional Inputs:** Precipitation or other meteorological data from public datasets, if incorporated into the model.

### Data Split:

*   **Training/Validation:** April 2021 – September 2022
*   **Testing:** October 2022 – April 2023 (strictly isolated to prevent data leakage)

## Project Structure

The project is organized as follows to ensure clarity and reproducibility:

| Directory/File           | Description                                         |
| :----------------------- | :-------------------------------------------------- |
| `20380357/`              | Raw USGS and NWM forecast files for station 20380357|
| `21609641/`              | Raw USGS and NWM forecast files for station 21609641|
| `data/processed/train/`  | Processed training data                             |
| `data/processed/val/`    | Processed validation data                           |
| `data/processed/test/`   | Processed test data                                 |
| `src/preprocess.py`      | Script for data cleaning and preparation            |
| `src/model.py`           | Defines the deep learning model architecture        |
| `src/train.py`           | Handles model training and validation               |
| `src/evaluate.py`        | Evaluates model performance and generates plots     |
| `src/utils.py`           | Utility functions for data handling and metrics     |
| `models/`                | Stores trained model files                          |
| `results/plots/`         | Saves box-plots and metric plots                    |
| `results/metrics/`       | Saves evaluation metric tables                      |
| `presentation/`          | Slideshow presentation for class presentation (10–15 min) covering approach, results, and future directions |
| `report/`                | Technical report (IEEE-style paper) detailing methodology, results, and implications, with link to private GitHub repo |
| `requirements.txt`       | Lists Python dependencies                           |
| `README.md`              | Project documentation (this file)                   |

## Dependencies

The project relies on the following Python packages, which can be installed using the provided `requirements.txt` using the Python interpreter's pip module:

*   **Python 3.10**: Core programming language (required for tensorflow-macos compatibility)
*   **TensorFlow or PyTorch**: Deep learning framework for model development
*   **Pandas**: Data manipulation and preprocessing
*   **NumPy**: Numerical computations
*   **Matplotlib**: Plotting runoff and metric visualizations
*   **Seaborn**: Enhanced visualization aesthetics
*   **Scikit-learn**: Evaluation metrics calculation
*   **Keras Tuner**: Hyperparameter optimization

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Data Preprocessing

The `src/preprocess.py` script performs the following tasks for each station:

1.  **Load Data:** Read NWM forecast and USGS observation CSV files.
2.  **Clean Data:** Handle missing values, convert units if necessary, and align timestamps between datasets.
3.  **Align Forecasts and Observations:** For each forecast cycle, match NWM forecasts with observed runoff at valid times (T + lead time).
4.  **Prepare Sequences:** Create input sequences including:
    *   Past observed runoff (e.g., last 24 hours).
    *   NWM forecasts for lead times 1–18 hours.
    *   Optional: Past precipitation data.
5.  **Split Data:** Organize into training, validation, and test sets, ensuring no test data (October 2022 – April 2023) is used during training.
6.  **Save Data:** Store processed datasets in `data/processed/` for model input.

## Model Architecture

*   **Model Choice:** A Long Short-Term Memory (`LSTM`) network is implemented due to its effectiveness in capturing temporal dependencies in time series data, ideal for forecasting runoff errors.
*   **Input Features:**
    *   Sequence of past observed runoff values (e.g., 24-hour window).
    *   NWM forecast values for lead times 1–18 hours.
    *   Optional: Sequence of past precipitation data.
*   **Output:** A vector of 18 predicted errors, one for each lead time.
*   **Alternative Architectures:** `GRU`, `Transformers`, or `CNN-LSTM` may be explored, with justification provided in the technical report.
*   **Activation Function:** [To be specified, e.g., `ReLU` for hidden layers, based on experimentation].
*   **Sequence-to-Sequence Setup:** The model handles multi-step forecasting by outputting corrections for all lead times simultaneously.

## Training

The `src/train.py` script manages model training with the following steps:

1.  **Load Data:** Retrieve processed training and validation datasets.
2.  **Define Model:** Implement the LSTM architecture in `src/model.py`.
3.  **Compile Model:** Use Mean Squared Error (`MSE`) loss and the `Adam` optimizer, selected for its adaptive learning rates suitable for time series data ([Adam Optimizer](https://arxiv.org/abs/1412.6980)).
4.  **Hyperparameter Tuning:** Employ `Keras Tuner` or Bayesian optimization to select optimal parameters, such as learning rate and number of LSTM units.
5.  **Regularization Techniques:**
    *   **Early Stopping:** Halt training if validation loss does not improve after a set number of epochs.
    *   **Dropout:** Apply dropout layers to prevent overfitting.
    *   **Batch Normalization:** Normalize layer inputs to stabilize training.
    *   **Gradient Clipping:** Mitigate exploding gradients, common in recurrent networks.
6.  **Learning Rate Scheduling:** Use `ReduceLROnPlateau` or `1 Cycle Scheduling` to adjust the learning rate dynamically.
7.  **Save Model:** Store the trained model in `models/` for evaluation.

## Evaluation

The `src/evaluate.py` script assesses model performance on the test set:

1.  **Load Model and Data:** Retrieve the trained model and test dataset.
2.  **Generate Corrected Forecasts:** Compute corrected forecasts by adding predicted errors to NWM forecasts.
3.  **Calculate Metrics:** For each lead time, compute:
    *   Coefficient of Correlation (`CC`)
    *   Root Mean Square Error (`RMSE`)
    *   Percent Bias (`PBIAS`)
    *   Nash-Sutcliffe Efficiency (`NSE`)
4.  **Visualizations:**
    *   **Runoff Box-Plots:** For each lead time, plot distributions of observed, NWM, and corrected runoff values across the test period.
    *   **Metric Plots:** Generate line plots of each metric versus lead time, comparing NWM and corrected forecasts.
5.  **Save Outputs:** Store plots in `results/plots/` and metric tables in `results/metrics/`.

## Results

The project produces the following outputs for each station:

*   **Runoff Box-Plots:** Three box-plots per lead time (1–18 hours) showing the distribution of observed, NWM, and corrected runoff values.
*   **Metric Plots:** Line plots for `CC`, `RMSE`, `PBIAS`, and `NSE` across lead times, comparing raw NWM and corrected forecasts.
*   **Metric Tables:** Detailed tables summarizing metric values, saved as CSV files.

## Usage Instructions

1.  **Organize Data:** Place raw NWM and USGS data in `data/raw/` as per the structure above.
2.  **Install Dependencies:** Run `pip install -r requirements.txt`.
3.  **Preprocess Data:** Execute `python src/preprocess.py` to prepare datasets.
4.  **Train Model:** Run `python src/train.py` to train the LSTM model.
5.  **Hyperparameter Tuning:** Execute `python src/tune.py --station <station>` to search optimal hyperparameters per station.
6.  **Evaluate Model:** Execute `python src/evaluate.py` to generate corrected forecasts, metrics, and plots.
7.  **Review Results:** Check `results/` for plots and metric files.

## Notes

*   **Data Leakage:** Ensure no test data (October 2022 – April 2023) is used during training or validation to maintain model integrity.
*   **Documentation:** All scripts should include detailed comments for reproducibility.
*   **Flexibility:** The framework supports alternative models (e.g., `GRU`, `Transformers`) and additional inputs (e.g., precipitation), with choices justified in the technical report.
*   **Station-Specific Models:** The project trains separate models for each station to account for unique catchment characteristics.

## References

*   Han, H. & Morrison, R. R. (2022). Improved runoff forecasting performance through error predictions using a deep-learning approach. *Journal of Hydrology*, *608*, 127653.
=======
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
>>>>>>> lstm_transformer

This project is based on methods from:
- Han, H. & Morrison, R. R. (2022). Improved runoff forecasting performance through error predictions using a deep-learning approach. Journal of Hydrology, 608, 127653.
