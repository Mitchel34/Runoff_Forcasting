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

**[PLOT: Insert map showing the two station locations with watershed boundaries]**
*Discussion points: These stations represent different hydrologic regimes and demonstrate the varying performance of NWM across different geographic regions.*

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

**[PLOT: Insert workflow diagram showing the pipeline from data input through model training to evaluation]**
*Discussion points: This end-to-end pipeline allows for systematic comparison between model types and provides a framework that could be extended to other stations.*

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

**[PLOT: Insert time series graph showing raw USGS observations vs NWM forecasts for a sample period]**
*Discussion points: Notice the systematic bias in NWM forecasts compared to observations - sometimes overestimating (Station 20380357) and sometimes closely following the pattern but with timing or magnitude errors (Station 21609641).*

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

**[PLOT: Insert diagram showing data structuring and sequence creation]**
*Discussion points: The sequence-based approach allows models to learn temporal patterns in NWM errors, and the structure ensures we're always predicting future errors rather than looking at contemporaneous data.*

---

## Slide 6: Model Architectures (1.5 mins)

*   **Goal:** Predict the 18 error values (error_1 to error_18) based on the past 24 hours of NWM flow and error features.
*   **For Each Station (21609641 & 20380357):**
    *   **LSTM Model:** Input (24x36) -> LSTM Layer -> Dense Output (18)
    *   **Transformer Model:** Input (24x36) -> Multiple Transformer Encoder Blocks (Multi-Head Attention + FeedForward) -> Global Average Pooling -> MLP Head -> Dense Output (18)
*   **Total Models Trained:** 4 (LSTM and Transformer for each station)

**[PLOT: Insert architecture diagrams for both LSTM and Transformer models]**
*Discussion points: LSTM captures sequential dependencies through its gates, while Transformer leverages self-attention to identify relationships between timesteps at any distance. The hyperparameters were tuned separately for each station to optimize performance.*

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

**[PLOT: Insert learning curves showing training and validation loss over epochs]**
*Discussion points: The learning curves demonstrate model convergence and help identify potential overfitting. The validation curves flatten after sufficient epochs, indicating that our early stopping criteria were effective.*

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

**[PLOT: Insert diagram illustrating the evaluation flow from prediction to metrics calculation]**
*Discussion points: These metrics were calculated for each lead time (1-18 hours), allowing us to assess how the models perform for different forecast horizons. This is crucial as forecasts typically degrade with increasing lead time.*

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

**[PLOT 1: Insert boxplot showing distribution of runoff values for Station 21609641 across lead times]**
*Discussion points: Note the long whiskers in the observed data boxplot, indicating high flow variability at this station. Both models (green) effectively match the observed flow distribution (blue), correcting the systematic biases in the original NWM forecast (orange).*

**[PLOT 2: Insert line graph showing NSE or RMSE metrics vs lead time for NWM and corrected forecasts]**
*Discussion points: The corrected forecasts maintain higher skill (NSE) or lower error (RMSE) than NWM across all lead times, with the greatest improvements at longer lead times where NWM accuracy naturally degrades.*

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

**[PLOT 1: Insert boxplot showing distribution of runoff values for Station 20380357 across lead times]**
*Discussion points: Notice the extremely short whiskers in the observed data boxplot, indicating very consistent low flows at this station (near 0-2 cfs). The NWM drastically overestimates flow (orange boxes up to 80+ cfs). Both models significantly reduced this overestimation but couldn't fully correct such a severe baseline error.*

**[PLOT 2: Insert monthly distribution boxplots for PBIAS or RMSE metrics]**
*Discussion points: The monthly distribution shows consistent NWM overestimation across all seasons. While our models reduced the bias, the fundamental disconnect between NWM and observations limited correction effectiveness.*

---

## Slide 11: Discussion

*   **Station Dependency:** Error correction success is highly dependent on the specific station's characteristics and the baseline NWM performance
*   **Baseline Quality Matters:** Deep learning models struggle to correct errors when the underlying physical model forecast (NWM) has very low initial skill
*   **Trade-offs:** LSTM and Transformer improved RMSE/PBIAS but not CC/NSE for the hardest station
*   **Limitations:** Negative NSE scores indicate corrected forecasts were still less accurate than simply using the mean observed flow for the test period

**[PLOT: Insert comparison plot showing improvement percentages across all metrics and both stations]**
*Discussion points: This comparison highlights the stark difference in correction success between stations. Where NWM had reasonable baseline performance (Station 21609641), our models achieved significant improvements. Where NWM fundamentally misrepresented the flow regime (Station 20380357), even sophisticated deep learning struggled to fully bridge the gap.*

---

## Slide 12: Future Directions & Improvements

*   More data (longer training periods)
*   Feature engineering (add precipitation, upstream NWM, catchment attributes)
*   Alternative models (CNN-LSTM, other Transformer variants)
*   Ensemble methods
*   Station-specific tuning
*   Investigate scaling warning
*   Post-processing (e.g., non-negative flow constraints)

**[PLOT: Insert conceptual diagram showing potential enhanced model architecture with additional data sources]**
*Discussion points: Future work could leverage additional meteorological and physiographic inputs that might help explain the systematic errors in NWM forecasts, particularly for challenging stations like 20380357.*

---

## Slide 13: Conclusion

*   Developed and evaluated LSTM and Transformer models to correct NWM runoff forecast errors for two USGS stations
*   **Trained and compared four models in total (LSTM and Transformer for each station)**
*   LSTM and Transformer showed promise for station 21609641, significantly reducing bias and magnitude errors
*   Both models struggled with station 20380357 due to poor baseline NWM forecast
*   The quality of the input physical model forecast is a critical limiting factor for data-driven error correction
*   Demonstrated a viable workflow for NWM error correction but highlighted the challenges and station-specific nature of the problem

**[PLOT: Insert a compelling "before and after" time series showing raw NWM, observations, and corrected forecast for a high-flow event]**
*Discussion points: This event illustrates how our error correction approach can lead to more accurate forecasts during critical high-flow periods, potentially improving flood prediction capabilities.*

---

## Slide 14: Thank You & Questions

*   **Acknowledgements:** (Optional)
*   **Questions?**
*   **(Visual):** Final compelling plot or workflow diagram
