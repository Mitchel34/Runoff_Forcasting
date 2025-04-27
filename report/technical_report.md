# Technical Report: Runoff Forecasting Error Correction

## 1. Introduction

[Brief overview of the project goals, motivation, and scope.]

## 2. Data Preprocessing

The initial phase involved processing raw National Water Model (NWM) forecast data and corresponding United States Geological Survey (USGS) stream gauge observations for two stations: 20380357 and 21609641. The raw data spanned from April 2021 to April 2023.

The preprocessing pipeline, implemented in `src/preprocess.py`, performed the following steps for each station:

1.  **Data Loading:** Monthly NWM forecast CSV files and the single USGS observation CSV file were loaded into pandas DataFrames. NWM files were concatenated.
2.  **Column Renaming and Lead Time Calculation:**
    *   Relevant columns in the NWM data (`streamflow_value`, `model_output_valid_time`, `model_initialization_time`) were renamed for clarity (to `nwm_flow`, `valid_time`, `init_time`).
    *   Timestamp columns were converted to datetime objects, explicitly handling the `YYYY-MM-DD_HH:MM:SS` format in NWM data.
    *   Lead time for each NWM forecast was calculated in hours as the difference between `valid_time` and `init_time`.
3.  **Lead Time Filtering:** NWM records were filtered to retain only those with lead times between 1 and 18 hours, inclusive, as required by the project scope.
4.  **USGS Data Handling:**
    *   The USGS data was loaded, parsing the `DateTime` column.
    *   The script identified the correct flow column (`USGSFlowValue` based on inspection during development).
    *   Unit conversion was applied to the USGS flow data, converting from cubic feet per second (cfs) to cubic meters per second (cms) by multiplying by 0.0283168, assuming the raw USGS data was in cfs. The identified flow column was renamed to `usgs_flow`.
    *   The USGS `DateTime` column was made timezone-naive (`tz_localize(None)`) to ensure compatibility with the NWM `valid_time` during merging.
5.  **Data Alignment and Merging:** The processed NWM and USGS DataFrames were merged based on the timestamp (`valid_time` from NWM matching `datetime` from USGS) using an inner join.
6.  **Error Calculation:** The forecast error, the primary target variable for the deep learning models, was calculated as `error = usgs_flow - nwm_flow`.
7.  **Data Pivoting:** The merged data, initially in a long format (one row per lead time per timestamp), was pivoted. The resulting DataFrame has a DateTime index, with columns representing `nwm_flow`, `usgs_flow`, and `error` for each of the 18 lead times (e.g., `nwm_flow_1`, `error_1`, `nwm_flow_2`, `error_2`, ..., `error_18`). Rows containing any NaN values after pivoting (indicating missing data for at least one lead time at that timestamp) were dropped.
8.  **Feature and Target Selection:**
    *   **Features (X):** Columns representing NWM flow and error for all 18 lead times (`nwm_flow_1` to `nwm_flow_18` and `error_1` to `error_18`) were selected as input features.
    *   **Targets (y):** Columns representing the error for lead times 1 through 18 (`error_1` to `error_18`) were selected as the prediction targets.
9.  **Sequence Creation:** The feature and target data were transformed into sequences suitable for time series models using a lookback window size of 24 hours.
    *   Input sequences (`X`) have the shape `(num_sequences, 24, 36)`, where 24 is the window size and 36 is the number of features (18 NWM flow + 18 error features).
    *   Target sequences (`y`) have the shape `(num_sequences, 18)`, representing the 18 error values to be predicted for the hour following the input sequence.
10. **Temporal Splitting:** The generated sequences were split chronologically into training and testing sets based on the timestamp corresponding to the end of each sequence:
    *   Training Set: April 2021 - September 30, 2022
    *   Testing Set: October 1, 2022 - April 2023
    This strict temporal split prevents data leakage from the test period into the training process.
11. **Data Scaling:** `StandardScaler` from scikit-learn was used to scale the data.
    *   An `x_scaler` was fitted *only* on the training features (`X_train`) and used to transform both `X_train` and `X_test`.
    *   A `y_scaler` was fitted *only* on the training targets (`y_train`) and used to transform both `y_train` and `y_test`.
12. **Saving Processed Data:** The scaled training and testing sequences (X and y) were saved as `.npz` files in `data/processed/train/` and `data/processed/test/`. The fitted `x_scaler` and `y_scaler` objects were saved using `joblib` to `data/processed/scalers/` for later use during evaluation and inference.

This preprocessing ensures that the data fed into the models is appropriately structured, scaled, and split for training and evaluation.

## 3. Model Development

Two distinct deep learning architectures were implemented, tailored to the characteristics of each station:

### 3.1 LSTM Model (Station 21609641)

Implemented in `src/models/lstm.py`, this model uses a Long Short-Term Memory (LSTM) network, suitable for capturing temporal dependencies in the more predictable runoff patterns of station 21609641.

-   **Framework:** TensorFlow/Keras
-   **Architecture:**
    -   Input Layer: Takes sequences of shape `(window_size, num_features)`, e.g., (24, 36).
    -   LSTM Layer: A single `tf.keras.layers.LSTM` layer processes the input sequence. The number of units (`lstm_units`) is a key hyperparameter. `return_sequences=False` is used, as only the final hidden state is needed to predict the errors for all subsequent lead times based on the input sequence.
    -   Output Layer: A `tf.keras.layers.Dense` layer maps the LSTM layer's output to the 18 target error values (one for each lead time from 1 to 18 hours).
-   **Function:** The `build_lstm_model` function constructs this Keras model.

### 3.2 Transformer Model (Station 20380357)

Implemented in `src/models/transformer.py`, this model utilizes a Transformer architecture, based on the "Attention Is All You Need" paper (Vaswani et al., 2017). This architecture is chosen for station 20380357 due to its potential to capture complex, non-sequential patterns often found in challenging hydrological regimes.

-   **Framework:** TensorFlow/Keras
-   **Architecture:**
    -   Input Layer: Takes sequences of shape `(window_size, num_features)`, e.g., (24, 36).
    -   Transformer Encoder Blocks: The core of the model consists of multiple stacked `transformer_encoder_block` units (number controlled by `num_encoder_blocks`). Each block contains:
        -   Multi-Head Self-Attention (`tf.keras.layers.MultiHeadAttention`): Allows the model to weigh the importance of different parts of the input sequence. Key hyperparameters include `num_heads` and `head_size`.
        -   Feed-Forward Network: Two dense layers with ReLU activation, controlled by `ff_dim`.
        -   Layer Normalization and Dropout: Applied within the block for stabilization and regularization (`dropout` hyperparameter).
        -   Residual Connections: Used around the attention and feed-forward sub-layers.
    -   Pooling Layer: `tf.keras.layers.GlobalAveragePooling1D` aggregates the outputs across the sequence dimension after the final encoder block.
    -   MLP Head: A final Multi-Layer Perceptron (consisting of Dense layers, configurable via `mlp_units` and `mlp_dropout`) maps the pooled representation to the 18 target error values.
-   **Function:** The `build_transformer_model` function constructs this Keras model.

## 4. Training and Tuning

The process of training the models and optimizing their hyperparameters is managed by dedicated scripts.

### 4.1 Utility Functions (`src/utils.py`)

This script centralizes helper functions, primarily the calculation of evaluation metrics required by the project:
-   `calculate_cc`: Pearson Correlation Coefficient.
-   `calculate_rmse`: Root Mean Square Error.
-   `calculate_pbias`: Percent Bias.
-   `calculate_nse`: Nash-Sutcliffe Efficiency.
These functions include basic handling for NaN values and potential division-by-zero errors.

### 4.2 Hyperparameter Tuning (`src/tune.py`)

This script automates the search for optimal hyperparameters using the Keras Tuner library.
-   **Objective:** Minimize validation loss (`mse`).
-   **Methodology:** Supports both `Hyperband` (for broad, fast exploration) and `BayesianOptimization` (for efficient convergence). A hybrid approach (Hyperband followed by Bayesian) is recommended.
-   **HyperModels:** Defines `LSTMHyperModel` and `TransformerHyperModel` classes that specify the model architecture and the search space for hyperparameters (e.g., `lstm_units`, `dropout_rate`, `learning_rate`, `num_encoder_blocks`, `num_heads`, `ff_dim`).
-   **Process:**
    1.  Loads training data and creates a time-series consistent validation split (using `train_test_split` with `shuffle=False`).
    2.  Instantiates the specified tuner (`Hyperband` or `BayesianOptimization`).
    3.  Runs the `tuner.search()` method, training multiple model configurations on the training data and evaluating them on the validation data.
    4.  Uses `EarlyStopping` within each trial to prevent wasted computation.
-   **Output:**
    1.  Prints a summary of the tuning results and the best hyperparameters found to the console.
    2.  Saves the best hyperparameters dictionary to a JSON file in the `results/hyperparameters/` directory (e.g., `results/hyperparameters/<station_id>_<model_type>_best_hps.json`). This file provides a persistent record and can be used to configure the final model training in `train.py`.

### 4.3 Model Training (`src/train.py`)

This script trains a single model instance using specified (or default/tuned) hyperparameters.
-   **Process:**
    1.  Loads preprocessed training data for the specified station.
    2.  Builds the specified model (LSTM or Transformer) using the provided hyperparameters.
    3.  Compiles the model with the Adam optimizer and Mean Squared Error (MSE) loss.
    4.  Sets up callbacks: `EarlyStopping` (monitors `val_loss`), `ModelCheckpoint` (saves the best model based on `val_loss` to `models/`), and `ReduceLROnPlateau`.
    5.  Trains the model using `model.fit()` with a `validation_split` of the training data.
-   **Usage:** Accepts command-line arguments for station, model type, epochs, batch size, learning rate, and model-specific hyperparameters (e.g., `--lstm_units`, `--tf_heads`). The best hyperparameters identified by `tune.py` should be passed here for final model training.

## 5. Evaluation

The performance of the trained models is assessed using the `src/evaluate.py` script on the held-out test set.

-   **Process:**
    1.  Loads the best trained model (`.keras` file) for the specified station and model type from the `models/` directory.
    2.  Loads the corresponding test data (`X_test`, `y_test_scaled`) and the essential metadata (`nwm_test_original`, `usgs_test_original`) saved during preprocessing.
    3.  Loads the `y_scaler` to inverse-transform predictions.
    4.  Generates error predictions using `model.predict()`.
    5.  Inverse-transforms the predicted errors.
    6.  Calculates the corrected NWM forecast: `Corrected = NWM_Original - Predicted_Error`.
    7.  For each lead time (1-18 hours):
        -   Calculates CC, RMSE, PBIAS, and NSE comparing USGS observations against both the original NWM forecast and the corrected forecast.
    8.  Saves the computed metrics to a CSV file in `results/metrics/`.
    9.  Generates and saves two box plots to `results/plots/`:
        -   Runoff Comparison: Observed vs. NWM vs. Corrected runoff per lead time.
        -   Metrics Comparison: NWM vs. Corrected metrics (CC, RMSE, PBIAS, NSE) per lead time.
-   **Usage:** Accepts command-line arguments for `station_id` and `model_type`.

## 6. Results and Discussion

[Analysis of results, comparison between stations/models.]

## 7. Conclusion and Future Work

[Summary, limitations, potential improvements.]
