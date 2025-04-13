Draft: Presentation Outline (10-15 mins)
(Assumes ~1 min per slide, adjust as needed)

Slide 1: Title Slide * Title: Improving NWM Runoff Forecasts with Seq2Seq LSTM Error Correction * Your Name(s) * Course Name/Number * Date

Slide 2: Motivation & Problem * Why care about runoff forecasting? (Floods, water resources - visuals helpful) * What is the National Water Model (NWM)? (Brief overview) * The Problem: NWM forecasts have errors, especially at longer lead times. (Show a simple plot illustrating NWM vs. Observed with errors).

Slide 3: Project Goal * Objective: Use a Deep Learning (Seq2Seq LSTM) model to predict and correct NWM forecast errors (1-18 hour leads) for two USGS stations. * Aim: Improve forecast accuracy compared to raw NWM and a simple baseline.

Slide 4: Data Overview * Sources: NWM Forecasts (hourly, 1-18h leads), USGS Observations (hourly). * Stations: 20380357, 21609641. * Period: Apr 2021 - Apr 2023. * Split: Train/Val (Apr 21 - Sep 22), Test (Oct 22 - Apr 23) - Emphasize NO data leakage. (Visual: Timeline).

Slide 5: Methodology: Preprocessing & Features * Key Steps: Load -> Align Time -> Calculate Errors (NWM - USGS) -> Scale Data. * Challenge: Aligning all 18 NWM leads with USGS observations (mention data loss ~75%). * Seq2Seq Input/Output Structure: (Use a simple diagram) * Encoder Input: Past 24h (Observed, NWM 1h, Error 1h) * Decoder Input: Current NWM Forecasts (Leads 1-18h) * Target: Actual Future Errors (Leads 1-18h)

Slide 6: Methodology: Models * Baseline: Simple Error Persistence. * Our Model: Seq2Seq LSTM (Encoder-Decoder). (Simple diagram showing LSTM blocks). Trained to predict the 18-hour error sequence.

Slide 7: Methodology: Training & Evaluation * Hyperparameter Tuning: keras_tuner with TimeSeriesSplit (robust validation). * Evaluation Metrics: CC, RMSE, PBIAS, NSE (explain briefly what 1-2 key ones like RMSE/NSE mean). * Comparison: Seq2Seq vs. Raw NWM vs. Baseline on the Test Set.

Slide 8: Results: Runoff Comparison * Show the Box-plot of Runoff (Observed vs. NWM vs. Baseline vs. Seq2Seq). * Point out how Seq2Seq corrected forecasts align better with Observed, especially compared to raw NWM. Highlight specific lead times if trends are clear.

Slide 9: Results: Metric Comparison (RMSE) * Show the Box-plot for RMSE vs. Lead Time. * Explain: Lower RMSE is better. Show how Seq2Seq consistently achieves lower RMSE than NWM and Baseline across most lead times.

Slide 10: Results: Metric Comparison (NSE) * Show the Box-plot for NSE vs. Lead Time. * Explain: NSE closer to 1 is better. Show improvement from Seq2Seq, especially where NWM/Baseline might be poor (NSE < 0).

Slide 11: Discussion & Limitations * Key Finding: Seq2Seq LSTM significantly improves NWM forecast accuracy by correcting errors. Outperforms simple baseline. * Limitations: * Data loss during alignment (~75%). * Only tested on two stations. * Model complexity.

Slide 12: Conclusion & Future Work * Conclusion: DL post-processing is a promising approach for enhancing operational hydrologic models like NWM. * Future Work: Test more stations, add features (precipitation), explore other models (Transformers?), imputation.

Slide 13: Thank You & Questions * Acknowledgements (if any). * Link to GitHub Repository. * "Questions?"