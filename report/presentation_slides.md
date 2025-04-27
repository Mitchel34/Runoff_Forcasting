\
# Presentation Outline: Runoff Forecasting Error Correction

**Target Duration:** 10-15 minutes

---

## Slide 1: Title Slide

*   **Title:** Correcting National Water Model Errors using Deep Learning for Improved Runoff Forecasting
*   **Group Members:** [List Names]
*   **Course:** CS 4440 AI
*   **Date:** April 27, 2025
*   **(Optional):** University Logo / Course Logo

---

## Slide 2: Introduction & Motivation (1 min)

*   **Problem:** The National Water Model (NWM) provides valuable nationwide runoff forecasts, but often contains significant errors (bias, magnitude inaccuracies) compared to observed stream gauge data (USGS).
*   **Goal:** Develop and evaluate deep learning models (LSTM, Transformer) to predict and correct these NWM errors for specific USGS stations.
*   **Motivation:** Accurate short-term runoff forecasts are crucial for water resource management, flood prediction, and ecological monitoring. Correcting NWM errors can lead to more reliable operational forecasts.
*   **Stations Studied:** 21609641 (California) & 20380357 (Colorado)

---

## Slide 3: Approach & Workflow (1 min)

*   **Overall Workflow:**
    1.  **Data Acquisition:** Gather raw NWM forecast data and USGS observations.
    2.  **Preprocessing:** Clean, align, calculate errors, create sequences, scale data.
    3.  **Model Development:** Implement LSTM and Transformer architectures.
    4.  **Hyperparameter Tuning:** Optimize model configurations using Keras Tuner.
    5.  **Training:** Train final models on the training dataset.
    6.  **Evaluation:** Assess model performance on a held-out test set using standard metrics (CC, RMSE, PBIAS, NSE) and compare against the original NWM forecast.
*   **(Visual):** High-level flowchart diagram illustrating these steps.

    ```mermaid
    graph TD
        A[Raw Data (NWM, USGS)] --> B(Preprocessing);
        B --> C{Model Selection};
        C -- Station 21609641 --> D(LSTM);
        C -- Station 20380357 --> E(Transformer);
        D --> F(Hyperparameter Tuning);
        E --> F;
        F --> G(Training);
        G --> H(Evaluation);
        H --> I[Results & Analysis];
    ```

---

## Slide 4: Data Sources (1 min)

*   **National Water Model (NWM):**
    *   Source: NOAA / National Weather Service
    *   Type: Hourly forecasts (we used v2.1)
    *   Features Used: Streamflow forecast (`streamflow_value`), initialization time, valid time.
*   **United States Geological Survey (USGS):**
    *   Source: USGS National Water Information System (NWIS)
    *   Type: Observed streamflow measurements.
    *   Features Used: Streamflow (`USGSFlowValue`), Timestamp.
*   **Time Period:** April 2021 - April 2023
*   **Lead Times:** Focused on 1 to 18 hours ahead.
*   **(Visual):** Map showing station locations OR sample NWM/USGS data plots.
    *   `[Screenshot: Map of station locations or sample time series plot from runoff_forecasting_models.ipynb]`

---

## Slide 5: Preprocessing Pipeline (1.5 mins)

*   **Key Steps (`src/preprocess.py`):**
    1.  Load monthly NWM files & single USGS file per station.
    2.  Calculate NWM lead times (Valid Time - Init Time).
    3.  Filter NWM for lead times 1-18h.
    4.  Convert USGS flow units (cfs -> cms) and align timezones.
    5.  Merge NWM & USGS on timestamp (inner join).
    6.  **Calculate Target Error:** `Error = Observed (USGS) - Forecast (NWM)`
    7.  Pivot data: Rows = Timestamps, Columns = `nwm_flow_1..18`, `usgs_flow_1..18`, `error_1..18`.
    8.  Create Sequences: Input `X` (24hr lookback window of NWM flow & error features), Target `y` (next hour's 18 error values).
    9.  Temporal Split: Train (Apr 2021 - Sep 2022), Test (Oct 2022 - Apr 2023).
    10. Scale Data: `StandardScaler` (fit *only* on training data).
    11. Save: Scaled data, original test flows, test timestamps, scalers.
*   **(Visual):** Code snippet from `src/preprocess.py` showing error calculation or sequence creation.
    *   `[Screenshot: Snippet from src/preprocess.py]`

---

## Slide 6: Model Architectures (1.5 mins)

*   **Goal:** Predict the 18 error values (`error_1` to `error_18`) based on the past 24 hours of NWM flow and error features.
*   **Station 21609641: LSTM (`src/models/lstm.py`)**
    *   Why LSTM? Good for capturing temporal dependencies, potentially suitable for stations with more predictable patterns.
    *   Architecture: Input (24x36) -> LSTM Layer -> Dense Output (18).
*   **Station 20380357: Transformer (`src/models/transformer.py`)**
    *   Why Transformer? Ability to capture complex, long-range dependencies and non-sequential patterns via self-attention; potentially better for more challenging hydrological regimes.
    *   Architecture: Input (24x36) -> Multiple Transformer Encoder Blocks (Multi-Head Attention + FeedForward) -> Global Average Pooling -> MLP Head -> Dense Output (18).
*   **(Visual):** Simplified diagrams of LSTM and Transformer architectures OR code snippets showing model definitions.
    *   `[Screenshot: tf.keras.Model summary or code snippet from src/models/lstm.py]`
    *   `[Screenshot: tf.keras.Model summary or code snippet from src/models/transformer.py]`

---

## Slide 7: Training & Tuning (1 min)

*   **Hyperparameter Tuning (`src/tune.py`):**
    *   Used Keras Tuner (`Hyperband` / `BayesianOptimization`).
    *   Objective: Minimize validation Mean Squared Error (MSE).
    *   Searched parameters like LSTM units, dropout, learning rate, Transformer heads/blocks/dims.
    *   Best hyperparameters saved to JSON.
*   **Final Training (`src/train.py`):**
    *   Used best hyperparameters found during tuning.
    *   Optimizer: Adam. Loss: MSE.
    *   Callbacks: `EarlyStopping` (prevent overfitting), `ModelCheckpoint` (save best model based on validation loss), `ReduceLROnPlateau`.
    *   Trained on the full training set with a validation split.
*   **(Visual):** Screenshot of Keras Tuner results OR terminal output during `src/train.py` execution showing epochs/loss.
    *   `[Screenshot: Keras Tuner best HPs output OR src/train.py terminal output]`

---

## Slide 8: Evaluation Methodology (1 min)

*   **Goal:** Assess if the "Corrected" forecast improves upon the original "NWM" forecast.
*   **Process (`src/evaluate.py`):**
    1.  Load best trained model and test data (including original NWM/USGS flows & timestamps).
    2.  Predict errors on the test set.
    3.  Inverse-transform predicted errors.
    4.  Calculate **Corrected Forecast = NWM Forecast + Predicted Error**.
    5.  Compare `Observed (USGS)` vs. `NWM Forecast` and `Observed (USGS)` vs. `Corrected Forecast`.
*   **Metrics:** Calculated per lead time (1-18h):
    *   CC (Correlation Coefficient): Phase agreement.
    *   RMSE (Root Mean Square Error): Magnitude error (lower is better).
    *   PBIAS (Percent Bias): Systematic over/underprediction (closer to 0 is better).
    *   NSE (Nash-Sutcliffe Efficiency): Overall model skill relative to mean observation (1 is perfect, <0 means mean is better).
*   **Analysis:**
    *   Overall metrics across the entire test period.
    *   Monthly metrics distribution (using box plots) to see performance consistency.
*   **(Visual):** Code snippet showing corrected forecast calculation OR table defining metrics.
    *   `[Screenshot: Snippet from src/evaluate.py showing corrected forecast calculation]`

---

## Slide 9: Key Results - Station 21609641 (LSTM) (1.5 mins)

*   **NWM Baseline:** High CC (>0.96), but severe overprediction (PBIAS ~1600-3200%), high RMSE (~11-21 cms), very poor NSE (<-500).
*   **LSTM Correction:**
    *   **Good:** Significantly reduced RMSE (~5-9 cms) and PBIAS (near 0%).
    *   **Bad:** Degraded CC (~0.2-0.3) and NSE remained poor (~-120 to -350, still < 0).
*   **Interpretation:** LSTM effectively corrected systematic bias and magnitude errors but disrupted the temporal correlation. Improved over NWM but still lacked overall skill (NSE < 0).
*   **(Visual):** Key result plots.
    *   `[Screenshot: results/plots/21609641_lstm_RMSE_lineplot.png]`
    *   `[Screenshot: results/plots/21609641_lstm_PBIAS_lineplot.png]`
    *   `(Optional) [Screenshot: results/plots/21609641_lstm_NSE_lineplot.png]`
    *   `(Optional) [Screenshot: results/plots/21609641_lstm_RMSE_distribution_boxplot.png]`

---

## Slide 10: Key Results - Station 20380357 (Transformer) (1.5 mins)

*   **NWM Baseline:** Extremely poor. Near-zero CC, massive RMSE (~9-97 cms), astronomical PBIAS, catastrophic NSE (millions negative). Essentially unusable.
*   **Transformer Correction:**
    *   Minimal improvement. RMSE reduced but still high (~4-29 cms).
    *   PBIAS remained erratic and large.
    *   NSE remained extremely negative.
    *   CC remained near zero.
*   **Interpretation:** The Transformer model struggled significantly. The baseline NWM forecast quality was too poor for the model to learn effective error correction patterns.
*   **(Visual):** Key result plots.
    *   `[Screenshot: results/plots/20380357_transformer_RMSE_lineplot.png]`
    *   `[Screenshot: results/plots/20380357_transformer_NSE_lineplot.png]`
    *   `(Optional) [Screenshot: results/plots/20380357_transformer_PBIAS_lineplot.png]`

---

## Slide 11: Discussion (1 min)

*   **Station Dependency:** Error correction success is highly dependent on the specific station's characteristics and the baseline NWM performance.
*   **Baseline Quality Matters:** Deep learning models struggle to correct errors when the underlying physical model forecast (NWM) has very low initial skill (e.g., near-zero correlation, extreme bias). "Garbage in, garbage out" applies.
*   **Trade-offs:** Observed a trade-off for the LSTM model (improved RMSE/PBIAS vs. degraded CC/NSE). Error correction isn't always a universal improvement across all metrics.
*   **Limitations:** Negative NSE scores indicate corrected forecasts were still less accurate than simply using the mean observed flow for the test period. The scaling warning noted in the report suggests minor caution about absolute metric values.

---

## Slide 12: Future Directions & Improvements (1 min)

*   **More Data:** Longer training periods might improve model robustness.
*   **Feature Engineering:** Incorporate additional relevant inputs (e.g., precipitation forecasts, upstream NWM forecasts, static catchment attributes).
*   **Alternative Models:** Explore other architectures (e.g., CNN-LSTM hybrids, different Transformer variants).
*   **Ensemble Methods:** Combine predictions from multiple models.
*   **Station-Specific Tuning:** More in-depth tuning for the challenging station (20380357).
*   **Investigate Scaling Warning:** Further debug the minor discrepancy observed between scaled/unscaled errors.
*   **Post-processing:** Apply constraints to ensure corrected forecasts remain physically plausible (e.g., non-negative flow).

---

## Slide 13: Conclusion (1 min)

*   **Summary:** We developed and evaluated LSTM and Transformer models to correct NWM runoff forecast errors for two USGS stations.
*   **Key Findings:**
    *   LSTM showed promise for Station 21609641, significantly reducing bias and magnitude errors, but at the cost of correlation.
    *   Transformer struggled with Station 20380357 due to the extremely poor quality of the baseline NWM forecast.
    *   The quality of the input physical model forecast is a critical limiting factor for data-driven error correction.
*   **Overall:** Demonstrated a viable workflow for NWM error correction but highlighted the challenges and station-specific nature of the problem.

---

## Slide 14: Thank You & Questions

*   **Acknowledgements:** (Optional: Mention any specific help or resources)
*   **Questions?**
*   **(Visual):** Maybe a final compelling plot or the workflow diagram again.

