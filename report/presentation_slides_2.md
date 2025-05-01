# Presentation Outline: Runoff Forecasting Error Correction

**Target Duration:** 10-15 minutes

---

## Slide 1: Title Slide

*   **Title:** Correcting National Water Model Errors using Deep Learning for Improved Runoff Forecasting
*   **Group Members:** [List Names]
*   **Course:** CS 4440 AI
*   **Date:** April 27, 2025

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
    3.  **Model Development:** Implement and train both LSTM and Transformer architectures for each station (total of 4 models).
    4.  **Hyperparameter Tuning:** Optimize model configurations using Keras Tuner.
    5.  **Training:** Train final models on the training dataset.
    6.  **Evaluation:** Assess model performance on a held-out test set using standard metrics (CC, RMSE, PBIAS, NSE) and compare against the original NWM forecast.
*   **(Visual):** High-level flowchart diagram illustrating these steps.

---

## Slide 4: Data Sources (1 min)

*   **National Water Model (NWM):**
    *   Source: NOAA / National Weather Service
    *   Type: Hourly forecasts (v2.1)
    *   Features: Streamflow forecast, initialization time, valid time
*   **USGS:**
    *   Source: USGS National Water Information System (NWIS)
    *   Type: Observed streamflow measurements
    *   Features: Streamflow, Timestamp
*   **Time Period:** April 2021 - April 2023
*   **Lead Times:** 1 to 18 hours ahead

---

## Slide 5: Preprocessing Pipeline (1.5 mins)

*   **Key Steps:**
    1.  Load monthly NWM files & single USGS file per station
    2.  Calculate NWM lead times (Valid Time - Init Time)
    3.  Filter NWM for lead times 1-18h
    4.  Convert USGS flow units (cfs -> cms) and align timezones
    5.  Merge NWM & USGS on timestamp (inner join)
    6.  **Calculate Target Error:** Error = Observed (USGS) - Forecast (NWM)
    7.  Pivot data: Rows = Timestamps, Columns = nwm_flow_1..18, usgs_flow_1..18, error_1..18
    8.  Create Sequences: Input X (24hr lookback window of NWM flow & error features), Target y (next hour's 18 error values)
    9.  Temporal Split: Train (Apr 2021 - Sep 2022), Test (Oct 2022 - Apr 2023)
    10. Scale Data: StandardScaler (fit only on training data)
    11. Save: Scaled data, original test flows, test timestamps, scalers

---

## Slide 6: Model Architectures (1.5 mins)

*   **Goal:** Predict the 18 error values (error_1 to error_18) based on the past 24 hours of NWM flow and error features.
*   **For Each Station (21609641 & 20380357):**
    *   **LSTM Model:** Input (24x36) -> LSTM Layer -> Dense Output (18)
    *   **Transformer Model:** Input (24x36) -> Multiple Transformer Encoder Blocks (Multi-Head Attention + FeedForward) -> Global Average Pooling -> MLP Head -> Dense Output (18)
*   **Total Models Trained:** 4 (LSTM and Transformer for each station)

---

## Slide 7: Training & Tuning (1 min)

*   **Hyperparameter Tuning:**
    *   Used Keras Tuner (Hyperband & BayesianOptimization)
    *   Objective: Minimize validation MSE
    *   Searched LSTM units, dropout, learning rate, Transformer heads/blocks/dims
    *   Best hyperparameters saved to JSON
*   **Final Training:**
    *   Used best hyperparameters
    *   Optimizer: Adam, Loss: MSE
    *   Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    *   Trained on full training set with validation split
    *   **All 4 models (LSTM and Transformer for both stations) were trained and saved**

---

## Slide 8: Evaluation Methodology (1 min)

*   **Goal:** Assess if the "Corrected" forecast improves upon the original "NWM" forecast.
*   **Process:**
    1.  Load best trained model and test data
    2.  Predict errors on the test set
    3.  Inverse-transform predicted errors
    4.  Calculate Corrected Forecast = NWM Forecast + Predicted Error
    5.  Compare Observed (USGS) vs. NWM Forecast and Observed (USGS) vs. Corrected Forecast
*   **Metrics:**
    *   CC (Correlation Coefficient)
    *   RMSE (Root Mean Square Error)
    *   PBIAS (Percent Bias)
    *   NSE (Nash-Sutcliffe Efficiency)

---

## Slide 9: Key Results - Station 21609641

*   **LSTM Model:**
    *   Maintained high CC (0.993-0.994)
    *   RMSE slightly higher at short leads, improved at long leads
    *   PBIAS improved to near zero (-1.6% to -4.3%)
    *   NSE improved, especially at longer leads (0.88-0.82)
*   **Transformer Model:**
    *   Also maintained high CC and improved PBIAS and NSE
    *   Performance similar to LSTM, with some differences in RMSE and bias correction
*   **Interpretation:** Both models effectively corrected systematic bias and improved skill, especially at longer lead times

---

## Slide 10: Key Results - Station 20380357

*   **LSTM Model:**
    *   RMSE reduced but still high
    *   PBIAS reduced but remained large
    *   NSE remained very negative
    *   CC remained near zero
*   **Transformer Model:**
    *   Slightly better RMSE reduction than LSTM
    *   PBIAS and NSE still poor, but some improvement
    *   CC remained near zero
*   **Interpretation:** Both models struggled due to poor baseline NWM forecast. Some improvement in magnitude, but overall skill still lacking

---

## Slide 11: Discussion

*   **Station Dependency:** Error correction success is highly dependent on the specific station's characteristics and the baseline NWM performance
*   **Baseline Quality Matters:** Deep learning models struggle to correct errors when the underlying physical model forecast (NWM) has very low initial skill
*   **Trade-offs:** LSTM and Transformer improved RMSE/PBIAS but not CC/NSE for the hardest station
*   **Limitations:** Negative NSE scores indicate corrected forecasts were still less accurate than simply using the mean observed flow for the test period

---

## Slide 12: Future Directions & Improvements

*   More data (longer training periods)
*   Feature engineering (add precipitation, upstream NWM, catchment attributes)
*   Alternative models (CNN-LSTM, other Transformer variants)
*   Ensemble methods
*   Station-specific tuning
*   Investigate scaling warning
*   Post-processing (e.g., non-negative flow constraints)

---

## Slide 13: Conclusion

*   Developed and evaluated LSTM and Transformer models to correct NWM runoff forecast errors for two USGS stations
*   **Trained and compared four models in total (LSTM and Transformer for each station)**
*   LSTM and Transformer showed promise for station 21609641, significantly reducing bias and magnitude errors
*   Both models struggled with station 20380357 due to poor baseline NWM forecast
*   The quality of the input physical model forecast is a critical limiting factor for data-driven error correction
*   Demonstrated a viable workflow for NWM error correction but highlighted the challenges and station-specific nature of the problem

---

## Slide 14: Thank You & Questions

*   **Acknowledgements:** (Optional)
*   **Questions?**
*   **(Visual):** Final compelling plot or workflow diagram
