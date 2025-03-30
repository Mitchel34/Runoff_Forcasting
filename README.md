# CS 4440 Artificial Intelligence
# Dr. Mohammad Ali Javidiian 
# Appalachian State University

# Final Project
# Mitchel Carson & Christian Castaneda Reneau
---------------------------------------------------------------------

## Project: Improving NWM Forecasts Using Deep Learning Post-processing

This repository contains the code and documentation for an AI course project aimed at improving short-range runoff forecasts from the National Water Model (NWM) using deep learning post-processing techniques. The project leverages Python and popular libraries to preprocess data, train a deep learning model, and evaluate the corrected forecasts against observed runoff data.

# Quick Start Guide
pip install -r requirements.txt
cd src
python preprocess.py
python model.py
python evaluate.py

The results will be saved in the results directory, including model files, evaluation metrics, and visualization plots.

### Process Description

This section describes the step-by-step workflow implemented in this project to preprocess data, develop a deep learning model, and evaluate the results as per the project requirements.

#### 1. Environment Setup
- **Prerequisites**: Ensure you have Python 3.8+ installed.
- **Dependencies**: Install required libraries using the provided `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```
  Key libraries include:
  - **Pandas**: For data loading and preprocessing.
  - **NumPy**: For numerical operations and array handling.
  - **Matplotlib** and **Seaborn**: For generating box-plots and visualizations.
  - **TensorFlow** or **PyTorch**: For building and training the deep learning model (choose one based on implementation).
  - **scikit-learn**: For computing evaluation metrics and data splitting.

#### 2. Data Preprocessing
- **Data Sources**:
  - NWM forecasts (hourly, lead times 1-18 hours) for two US stations (April 2021 - April 2023).
  - USGS observed runoff data (hourly, same period).
  - Optional: Precipitation data (e.g., from NOAA).
- **Steps**:
  1. Load data using Pandas:
     ```python
     import pandas as pd
     nwm_data = pd.read_csv('path_to_nwm_forecasts.csv')
     usgs_data = pd.read_csv('path_to_usgs_observations.csv')
     ```
  2. Clean the data:
     - Handle missing values (e.g., interpolate or drop).
     - Align timestamps between NWM and USGS datasets.
     - Convert units if necessary (e.g., runoff from cfs to m³/s if required).
  3. Split the data:
     - Training/Validation: April 2021 - September 2022.
     - Testing: October 2022 - April 2023.
     ```python
     train_val_data = nwm_data[nwm_data['date'] < '2022-10-01']
     test_data = nwm_data[nwm_data['date'] >= '2022-10-01']
     ```
  4. Normalize features (e.g., runoff, precipitation) using scikit-learn:
     ```python
     from sklearn.preprocessing import StandardScaler
     scaler = StandardScaler()
     train_val_data_scaled = scaler.fit_transform(train_val_data[['runoff']])
     ```

#### 3. Model Development
- **Architecture**: We implemented a Long Short-Term Memory (LSTM) network using TensorFlow (or PyTorch) to predict residuals (errors) between NWM forecasts and USGS observations. Alternatives like GRU or Transformers can be explored.
- **Steps**:
  1. Prepare input sequences:
     - Use a sliding window approach to create sequences of past NWM forecasts, observed runoff, and optional precipitation data (e.g., past 24 hours) to predict residuals for lead times 1-18 hours.
     ```python
     import numpy as np
     def create_sequences(data, seq_length, lead_times):
         X, y = [], []
         for i in range(len(data) - seq_length - max(lead_times)):
             X.append(data[i:i+seq_length])
             y.append(data[i+seq_length:i+seq_length+max(lead_times)])
         return np.array(X), np.array(y)
     seq_length = 24
     X_train, y_train = create_sequences(train_val_data_scaled, seq_length, range(1, 19))
     ```
  2. Build the model (example using TensorFlow):
     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import LSTM, Dense
     model = Sequential([
         LSTM(64, return_sequences=True, input_shape=(seq_length, num_features)),
         LSTM(32),
         Dense(18)  # Output residuals for 18 lead times
     ])
     model.compile(optimizer='adam', loss='mse')
     ```
  3. Train the model:
     - Split training/validation data (e.g., 80/20 split within April 2021 - September 2022).
     - Ensure no test data leakage.
     ```python
     model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
     ```

#### 4. Model Testing and Analysis
- **Steps**:
  1. Generate predictions on the test set (October 2022 - April 2023):
     ```python
     X_test, _ = create_sequences(test_data_scaled, seq_length, range(1, 19))
     residuals_pred = model.predict(X_test)
     corrected_forecasts = test_data['nwm_runoff'][seq_length:] + residuals_pred
     ```
  2. Compare corrected forecasts with NWM forecasts and USGS observations.

#### 5. Evaluation and Visualization
- **Metrics**: Compute the following for each lead time (1-18 hours):
  - Coefficient of Correlation (CC)
  - Root Mean Square Error (RMSE)
  - Percent Bias (PBIAS)
  - Nash-Sutcliffe Efficiency (NSE)
  ```python
  from sklearn.metrics import mean_squared_error, r2_score
  import numpy as np
  def compute_metrics(obs, pred):
      cc = np.corrcoef(obs, pred)[0, 1]
      rmse = np.sqrt(mean_squared_error(obs, pred))
      pbias = 100 * (np.sum(pred - obs) / np.sum(obs))
      nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
      return cc, rmse, pbias, nse
  ```
- **Plots**:
  1. Box-plot of Observed, NWM, and Corrected Runoff:
     ```python
     import matplotlib.pyplot as plt
     import seaborn as sns
     data_to_plot = [usgs_test['runoff'], test_data['nwm_runoff'], corrected_forecasts]
     sns.boxplot(data=data_to_plot)
     plt.xticks([0, 1, 2], ['Observed', 'NWM', 'Corrected'])
     plt.show()
     ```
  2. Box-plots of Metrics:
     - Compute metrics for each lead time and plot using Seaborn.

#### 6. Running the Code
- **Directory Structure**:
  ```
  ├── data/                # NWM and USGS data files
  ├── src/                 # Source code
  │   ├── preprocess.py    # Data preprocessing script
  │   ├── model.py         # Model definition and training
  │   ├── evaluate.py      # Testing and evaluation
  ├── results/             # Output plots and metrics
  ├── requirements.txt     # Dependencies
  └── README.md            # This file
  ```
- **Execution**:
  1. Place data files in `data/` directory.
  2. Run preprocessing:
     ```bash
     python src/preprocess.py
     ```
  3. Train the model:
     ```bash
     python src/model.py
     ```
  4. Evaluate and generate plots:
     ```bash
     python src/evaluate.py
     ```

#### 7. Notes
- Ensure no test data (October 2022 - April 2023) is used during training or validation to avoid data leakage.
- Adjust hyperparameters (e.g., sequence length, LSTM units) in `model.py` as needed.
- Results are saved in the `results/` folder as plots and CSV files.

---

This README provides a clear, reproducible process for executing the project. You can adapt the specifics (e.g., model architecture, file paths) based on your actual implementation. Let me know if you'd like me to refine any section further!