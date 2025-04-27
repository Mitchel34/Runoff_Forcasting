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

[Details about LSTM and Transformer architectures.]

## 4. Training and Tuning

[Details about training process, hyperparameters, tuning.]

## 5. Evaluation

[Metrics, visualizations, comparison.]

## 6. Results and Discussion

[Analysis of results, comparison between stations/models.]

## 7. Conclusion and Future Work

[Summary, limitations, potential improvements.]
