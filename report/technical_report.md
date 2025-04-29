# Technical Report: Runoff Forecasting Error Correction

## 1. Introduction

Accurate streamflow forecasting is essential for water resource management, flood prediction, and ecological assessment. While the National Water Model (NWM) provides operational forecasts across the United States, it often exhibits systematic biases and errors that vary by location and lead time. This project develops and evaluates deep learning approaches to correct NWM forecast errors for two USGS gauge stations with distinct hydrological characteristics.

The project goals are to:
1. Develop station-specific deep learning models (LSTM and Transformer architectures) to predict and correct NWM forecast errors
2. Compare model performance across different watershed types and forecasting lead times
3. Evaluate the effectiveness of error correction in improving streamflow prediction accuracy

The scope encompasses 18-hour forecasting horizons for two contrasting watersheds: station 21609641 with relatively predictable hydrology and station 20380357 with challenging hydrological conditions.

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
    *   **No unit conversion is applied.** The project assumes all flow data (NWM and USGS) is in **cubic feet per second (cfs)** and maintains this unit throughout. The identified flow column was renamed to `usgs_flow`.
    *   The USGS data was resampled to an hourly frequency using the mean (`.resample('H').mean()`) and missing values were filled using linear interpolation (`.interpolate(method='linear')`) followed by backfill/forward fill.
    *   The USGS `DateTime` column was made timezone-naive (`tz_localize(None)`) to ensure compatibility with the NWM `valid_time` during merging.
5.  **Data Alignment and Merging:** The processed NWM and USGS DataFrames were merged based on the timestamp (`valid_time` from NWM matching `datetime` from USGS) using an inner join.
6.  **Error Calculation:** The forecast error, the primary target variable for the deep learning models, was calculated as `error = usgs_flow - nwm_flow` (in cfs).
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
12. **Saving Processed Data:** The scaled training and testing sequences (X and y), along with the original NWM and USGS flow values corresponding to the test set targets (`nwm_test_original`, `usgs_test_original`), and the timestamps for the test sequences (`test_timestamps`), were saved as `.npz` files in `data/processed/train/` and `data/processed/test/`. The fitted `x_scaler` and `y_scaler` objects were saved using `joblib` to `data/processed/scalers/` for later use during evaluation and inference.

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
    2.  Loads the corresponding test data (`X_test`, `y_test_scaled`) and the essential metadata (`nwm_test_original`, `usgs_test_original`, `test_timestamps`) saved during preprocessing.
    3.  Loads the `y_scaler` to inverse-transform predictions.
    4.  Generates error predictions using `model.predict()`.
    5.  Inverse-transforms the predicted errors (`predicted_errors_unscaled`).
    6.  Calculates the corrected NWM forecast using the formula: `Corrected = NWM_Original + Predicted_Error`. This formula is used because the error was defined as `USGS - NWM` during preprocessing, so adding the predicted error to the NWM forecast aims to bring it closer to the observed USGS value.
    7.  Calculates overall evaluation metrics (CC, RMSE, PBIAS, NSE) for each lead time (1-18 hours) by comparing USGS observations against both the original NWM forecast and the corrected forecast.
    8.  Saves the computed overall metrics to a CSV file in `results/metrics/`.
    9.  Calculates monthly evaluation metrics (CC, RMSE, PBIAS, NSE) for each lead time by grouping the test set results by month using the `test_timestamps`. This provides insight into the variability of performance over the test period.
    10. Generates and saves plots to `results/plots/`:
        -   Runoff Comparison Box Plot: Observed vs. NWM vs. Corrected runoff per lead time (e.g., `*_runoff_boxplot.png`).
        -   Overall Metrics Line Plots: NWM vs. Corrected metrics (CC, RMSE, PBIAS, NSE) per lead time (e.g., `*_CC_lineplot.png`).
        -   Monthly Metric Distribution Box Plots: Distribution of monthly CC, RMSE, PBIAS, and NSE values for NWM vs. Corrected forecasts across lead times (e.g., `*_RMSE_distribution_boxplot.png`).
-   **Usage:** Accepts command-line arguments for `station_id` and `model_type`.

## 6. Results and Discussion

Our evaluation assessed the performance of the trained error correction models against the original NWM forecasts using the test set (October 2022 - April 2023). The results show significant differences between stations and model architectures.

### 6.1 Station 21609641 Performance

For this station with more predictable hydrology:

- **Original NWM Performance**: The NWM consistently underestimated streamflow, with this bias becoming more pronounced at longer lead times. While correlation was high, the bias resulted in poor overall accuracy.

- **LSTM Model Performance**:
  - Successfully corrected the systematic underestimation bias across all lead times
  - Maintained consistent flow predictions that closely match observed values
  - Achieved excellent correlation with observed flows
  - Demonstrated stable performance even at longer lead times (14-18 hours)

- **Transformer Model Performance**:
  - Performed similarly well to the LSTM model
  - Showed slightly more consistent corrections at longer lead times (14-18 hours)
  - Effectively eliminated bias across the entire forecast horizon
  - Visual comparison shows remarkable alignment between corrected forecasts and observations

Both models successfully transformed the significantly biased NWM forecasts into accurate streamflow predictions for this station, with the Transformer showing marginally better performance at longer horizons.

### 6.2 Station 20380357 Performance

For this challenging watershed:

- **Original NWM Performance**: The NWM drastically overestimated flows by orders of magnitude, with observed flows near zero while forecasts ranged from 20-90 cfs. This extreme bias increased with lead time and rendered the raw forecasts essentially unusable.

- **LSTM Model Performance**:
  - Dramatically reduced forecast values from 40-80 cfs to near 0 cfs
  - Significantly improved PBIAS, reducing it from over 23,000% to under 100%
  - Substantially lowered RMSE values
  - However, still showed systematic overestimation, albeit much less severe than original NWM

- **Transformer Model Performance**:
  - Showed similar major improvements in reducing the extreme bias
  - Achieved slightly better RMSE reduction than the LSTM
  - Produced more balanced corrections across different flow ranges
  - Despite improvements, still struggled with capturing the precise dynamics of this watershed

While both models significantly improved forecasts for this challenging watershed, neither could fully overcome the fundamental difficulties presented by the extremely poor baseline NWM performance.

### 6.3 Comparison of Model Architectures

- **LSTM Strengths**: Performed well on systematic biases and showed good skill at capturing short-term patterns.

- **Transformer Strengths**: Demonstrated slightly more consistent performance across lead times and marginally better metrics for the challenging watershed.

- **Station Dependency**: Model architecture choice was less important than the inherent predictability of the watershed. Both architectures performed similarly for each respective station, suggesting that watershed characteristics, rather than model selection, were the dominant factor in forecast quality.

- **Lead Time Performance**: Both architectures showed degraded performance at longer lead times, but the decline was much more pronounced for station 20380357.

### 6.4 Key Observations

1. **Watershed-Dependent Efficacy**: Deep learning correction techniques are highly effective for watersheds where the NWM has moderate skill (station 21609641) but face limitations in watersheds where the NWM fundamentally lacks skill (station 20380357).

2. **Bias Correction**: Both models excel at correcting systematic bias, which was the dominant error component in the NWM forecasts for both stations.

3. **Hyperparameter Optimization Impact**: The tuning process revealed that transformers benefit from no dropout (0.0), while LSTMs require moderate dropout (0.2-0.3). All models converged to the same optimal learning rate (0.001).

4. **Architecture Adaptability**: Different watersheds benefit from different model configurations, with the transformer for station 20380357 requiring twice the attention heads (8 vs. 4) compared to station 21609641.

## 7. Conclusion and Future Work

This study demonstrates that deep learning approaches can substantially improve NWM streamflow forecasts by correcting systematic errors. Both LSTM and Transformer architectures showed significant skill in error correction, though their effectiveness was highly dependent on watershed characteristics and the quality of the underlying NWM forecasts.

### Key Findings:

1. For watersheds where the NWM shows moderate skill (station 21609641), error correction models can transform biased forecasts into highly accurate predictions across all lead times.

2. For challenging watersheds with poor NWM performance (station 20380357), deep learning models can dramatically reduce bias but may not fully overcome fundamental deficiencies in the baseline forecast.

3. The choice between LSTM and Transformer architectures has less impact than watershed characteristics, suggesting that site-specific factors are more critical than model architecture selection.

4. Hyperparameter optimization is crucial for maximizing model performance, with different optimal configurations emerging for each station-model combination.

### Limitations:

1. The correction models are station-specific and would require retraining for different locations.

2. When baseline NWM forecasts completely lack skill (e.g., station 20380357), even sophisticated deep learning approaches face fundamental limitations.

3. The current approach requires historical observations for model training, limiting application to gauged watersheds.

### Future Work:

1. **Regionalization**: Develop regional error correction models that can be applied to ungauged watersheds by leveraging watershed characteristics and nearby gauge performance.

2. **Multivariate Inputs**: Incorporate additional meteorological variables (precipitation, temperature) and watershed characteristics to improve prediction accuracy.

3. **Hybrid Physics-ML Models**: Explore integrating physical constraints with deep learning to improve performance in challenging watersheds.

4. **Ensemble Methods**: Implement ensemble approaches combining multiple error correction models to increase robustness and quantify uncertainty.

5. **Operational Integration**: Develop operational workflows for real-time error correction that can be integrated with existing NWM forecast systems.

This research demonstrates the potential for deep learning to enhance operational streamflow forecasts, particularly for correcting systematic biases. The station-dependent performance highlights the need for careful model selection and evaluation based on local watershed characteristics.
