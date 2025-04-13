# NWM Runoff Forecast Correction Project Structure

```
nwm_dl_postprocessing/
├── data/
│   ├── raw/
│   │   ├── 20380357/       # Stream: 20380357
│   │   └── 21609641/       # Stream: 21609641
│   └── processed/
│       ├── train_validation_data.csv   # Cleaned and aligned data for training and validation
│       └── test_data.csv               # Strictly held-out test data (October 2022–April 2023)
│
├── models/
│   └── nwm_lstm_model.keras            # Trained Seq2Seq LSTM model with optimized hyperparameters
│
├── notebooks/
│   ├── exploratory_analysis.ipynb      # Exploratory data analysis of NWM and USGS time series
│   ├── model_development.ipynb         # Prototyping Seq2Seq LSTM model with hyperparameter tuning
│   └── results_visualization.ipynb     # Generating plots and analyzing model evaluation metrics
│
├── src/
│   ├── preprocess.py                   # Data loading, cleaning, feature engineering, and splitting
│   ├── model.py                        # Seq2Seq LSTM model architecture with ReLU activation and Adam optimizer
│   ├── tuner.py                        # KerasTuner with TimeSeriesSplit for hyperparameter optimization
│   ├── baseline.py                     # Simple persistence-based baseline model for error correction
│   ├── predict.py                      # Generates runoff predictions from trained model
│   ├── evaluate.py                     # Calculates CC, RMSE, PBIAS, NSE metrics
│   └── visualize.py                    # Creates box plots using Seaborn and Matplotlib
│
├── tests/
│   ├── test_preprocess.py              # Unit tests for preprocessing and data integrity
│   └── test_model.py                   # Unit tests for model structure and tuning logic
│
├── reports/
│   ├── figures/
│   │   ├── runoff_boxplots.png         # Box plots: Observed vs NWM vs Corrected (LSTM) vs Corrected (Baseline) runoff
│   │   └── metrics_boxplots.png        # Box plots of evaluation metrics across lead times
│   └── technical_report.pdf            # Final report formatted for IEEE or hydrology publication
│
├── presentation/
│   ├── presentation_slides.pdf         # Final presentation slides
│   └── presentation_notes.md           # Talking points and presenter notes
│
├── requirements.txt                    # Python dependencies, including keras-tuner
├── README.md                           # Project summary, setup instructions, and usage guide
└── .gitignore                          # Files and directories to exclude from version control
```

---

## Dependencies (`requirements.txt`)
```plaintext
tensorflow>=2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
keras-tuner
jupyter
```

---

## Key Features and Considerations

### ✅ Clean Data Handling
- Data is separated into raw and processed directories to prevent contamination.
- Strict adherence to time-based train/validation (April 2021–September 2022) and test (October 2022–April 2023) splits.

### ✅ Model Architecture
- Uses a Sequence-to-Sequence LSTM neural network with ReLU activation functions and the Adam optimizer.
- Designed specifically to correct residuals between NWM forecasts and USGS observed runoff across all lead times simultaneously.

### ✅ Automated Hyperparameter Tuning
- Integrated `keras_tuner` via `tuner.py` to perform scalable searches using HyperBand or Bayesian optimization.
- Uses TimeSeriesSplit for robust time-series validation within the training period.
- Tunes number of LSTM units, learning rate, dropout, sequence length, and number of layers.

### ✅ Baseline Comparison
- Implements a simple persistence-based baseline model for error correction.
- Provides context for evaluating the added complexity versus performance gain of the Seq2Seq approach.

### ✅ Comprehensive Evaluation
- Hydrologic metrics computed across 1–18 hour lead times:
  - Coefficient of Correlation (CC)
  - Root Mean Square Error (RMSE)
  - Percent Bias (PBIAS)
  - Nash-Sutcliffe Efficiency (NSE)

### ✅ Visualization and Reporting
- High-quality box plots for forecast comparisons and metric distributions.
- Final results presented in both Jupyter Notebooks and static figures for reports.

### ✅ Reproducibility and Collaboration
- Full documentation in the README with steps to reproduce results.
- Modular scripts and unit tests enhance maintainability and ease of collaboration.

---

## Quickstart Guide

This guide will help you set up and run the NWM Runoff Forecast Correction project.

### Prerequisites
- Python 3.8+ installed
- Git (for cloning the repository)
- Sufficient disk space for data (~500MB)
- (Optional) CUDA-compatible GPU for faster model training

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/nwm-runoff-forecast-correction.git
cd nwm-runoff-forecast-correction
```

### 2. Set Up a Python Virtual Environment
```bash
# Create a virtual environment
python -m venv nwm_env

# Activate the virtual environment
# On Windows:
nwm_env\Scripts\activate
# On macOS/Linux:
source nwm_env/bin/activate
```

### 3. Install Dependencies
```bash
# Install required packages
pip install -r nwm_dl_postprocessing/requirements.txt
```

### 4. Prepare the Data
```bash
# Create the required directory structure if it doesn't exist
mkdir -p nwm_dl_postprocessing/data/raw/20380357
mkdir -p nwm_dl_postprocessing/data/raw/21609641
mkdir -p nwm_dl_postprocessing/data/processed

# Copy raw data files to their respective directories
cp -r 20380357/* nwm_dl_postprocessing/data/raw/20380357/
cp -r 21609641/* nwm_dl_postprocessing/data/raw/21609641/

# Process the raw data
python -m nwm_dl_postprocessing.src.preprocess
```

### 5. Explore the Data (Optional)
```bash
# Launch Jupyter to explore data and model development notebooks
cd nwm_dl_postprocessing
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### 6. Train the Model
```bash
# Option 1: Run the training script directly
python -m nwm_dl_postprocessing.src.model

# Option 2: Use the provided shell script (Unix/macOS)
bash nwm_dl_postprocessing/train_models.sh
```

### 7. Evaluate Model Performance
```bash
# Generate predictions on test set
python -m nwm_dl_postprocessing.src.predict

# Calculate evaluation metrics
python -m nwm_dl_postprocessing.src.evaluate

# Generate visualization plots
python -m nwm_dl_postprocessing.src.visualize
```

### 8. View Results
Results and visualizations will be saved in the `nwm_dl_postprocessing/reports/figures/` directory.

### Common Issues and Solutions

1. **Missing Data Files**: Ensure all raw data files are correctly placed in their respective directories.

2. **Memory Errors During Processing**: For large datasets, consider processing in smaller batches or use a machine with more RAM.

3. **CUDA/GPU Issues**: If encountering GPU-related errors, try forcing CPU usage by setting:
   ```bash
   export CUDA_VISIBLE_DEVICES=-1
   ```

4. **TimeSeriesSplit Warnings**: These are expected during hyperparameter tuning and can be safely ignored.

### Running Tests
```bash
# Run all tests
python -m unittest discover nwm_dl_postprocessing/tests

# Run specific test file
python -m unittest nwm_dl_postprocessing.tests.test_preprocess
```

For more detailed information, refer to the project structure and task sections below.

---

# Deep Learning for Improved Runoff Forecasting

## Introduction and Context
Accurate runoff forecasting is vital for flood prediction, water resource management, and hydrologic analyses. The United States National Water Model (NWM), developed by NOAA, provides short-range runoff forecasts across the continental US. However, these forecasts can exhibit systematic and time-dependent errors, especially at longer lead times. Recent advances in Deep Learning (DL) offer potential to enhance physically based hydrologic models by post-processing forecasts. Sequence models like Long Short-Term Memory (LSTM) networks can learn to predict NWM forecast errors, producing corrected runoff forecasts that align better with observed data. This project draws inspiration from Han & Morrison (2022), which combines NWM outputs with observed runoff and precipitation in a DL framework.

## Project Goal
The objective is to apply a Sequence-to-Sequence LSTM model to improve NWM short-range runoff forecasting by predicting forecast errors. The tasks include:
1.  **Preprocessing**: Prepare NWM forecasts and USGS observed runoff data for two US stations. Define input sequences using past observations and forecasts.
2.  **Model Development**: Train, validate, and test a single Seq2Seq DL model to predict NWM forecast errors for lead times of 1–18 hours simultaneously. Use robust time-series cross-validation during hyperparameter tuning.
3.  **Evaluation**: Assess corrected forecasts using standard hydrologic metrics and compare against original NWM forecasts and a simple baseline correction model.

## Data Description and Usage Rules
### Data Sources
-   **NWM Forecasts**: Hourly short-range forecasts (1–18 h lead times) for two US stations, spanning April 2021 to April 2023. Contained in monthly files (`streamflow_*.csv`).
-   **USGS Observations**: Hourly observed runoff data for the same period. Contained in `*_Strt_*.csv` files.
-   **Derived Data**: NWM forecast errors (residuals) calculated as `NWM_forecast - USGS_observation` for each lead time.

### Train/Validation/Test Split
-   **Training/Validation**: April 2021 – September 2022 (Used for model training and hyperparameter tuning with TimeSeriesSplit).
-   **Testing**: October 2022 – April 2023 (Strictly held-out set for final evaluation).

**Note**: Test set data (October 2022 – April 2023) must not be used during training or validation to prevent data leakage.

## Tasks
### 4.1 Data Preprocessing
-   Clean data (e.g., handle missing values, align time steps, convert units).
-   Calculate NWM forecast errors (residuals) for each lead time.
-   **Feature Engineering**: Construct sequences for the Seq2Seq model.
    -   **Encoder Input Sequence**: Use a fixed window (e.g., the past 24 hours) of:
        -   USGS observed runoff.
        -   NWM 1-hour lead forecasts.
        -   Calculated 1-hour lead forecast errors (NWM 1h forecast - USGS observation).
    -   **Decoder Input Sequence**: NWM forecasts for lead times 1-18 hours issued at the *current* time step (these are the forecasts needing correction).
    -   **Target Output Sequence**: The *actual* NWM forecast errors for lead times 1-18 hours over the *next* 18 hours.
-   Scale features appropriately (e.g., using `StandardScaler` fit only on the training data).
-   Split data into training+validation and testing sets per the specified time frames.

### 4.2 Model Development
-   **Architecture**: Implement a Sequence-to-Sequence (Seq2Seq) model using LSTM layers for both the encoder and decoder.
-   **Target**: Train the single model to predict the sequence of NWM forecast errors for lead times 1–18 hours simultaneously.
-   **Hyperparameter Tuning**: Use `keras_tuner` (e.g., HyperBand or BayesianOptimization) to optimize parameters like LSTM units, learning rate, dropout, etc.
    -   **Validation Strategy**: Employ `sklearn.model_selection.TimeSeriesSplit` within the KerasTuner search process on the training+validation dataset (April 2021 – September 2022) to ensure robust hyperparameter selection suitable for time-series data.
-   Ensure no future or test data is used during training or tuning.

### 4.3 Model Testing and Analysis
-   Evaluate the final tuned model on the held-out test set (October 2022 – April 2023).
-   Generate predicted error sequences for each time step in the test set.
-   Calculate corrected forecasts: `Corrected_Forecast[lead] = NWM_Forecast[lead] - Predicted_Error[lead]`.
-   **Baseline Comparison**: Implement and evaluate a simple baseline error correction model (e.g., persistence: predict error for lead `L` at time `t+L` as the observed error for lead `L` at time `t`).
-   Compare corrected forecasts against:
    -   Original NWM forecasts
    -   Observed USGS runoff
    -   Baseline corrected forecasts

## Required Results and Plots
1.  **Box-plot of Runoff**:
    -   Compare Observed (USGS), Forecasted (NWM), Corrected (Seq2Seq model), and Corrected (Baseline) runoff for each lead time (1–18 h).
    -   Display four box-plots per lead time.

2.  **Box-plots of Evaluation Metrics**:
    -   Compute and plot for each lead time (1–18 h): CC, RMSE, PBIAS, NSE.
    -   Present metrics as four box-plots (one per metric), each with 18 sets of boxes (one set per lead time).
    -   Compare metrics for the original NWM forecasts, the Seq2Seq corrected forecasts, and the Baseline corrected forecasts.

## Deliverables
### (a) Technical Report
- Format: Scientific paper style (e.g., IEEE or hydrology journal).
- Content:
  - **Methodology**: Data sources, preprocessing, Seq2Seq model architecture, and training details.
  - **Results**: Required plots, tables, error metrics, and discussion.
  - **Implications**: Compare model performance to NWM forecasts and baseline correction, discuss challenges and limitations.
- Include a link to a private GitHub repository with well-documented code (e.g., README.md with instructions to reproduce results).

### (b) Class Presentation
- Duration: 10–15 minutes.
- Content:
  - Approach and workflow
  - Key results and findings
  - Future directions
- All group members must participate.

## Important Notes
- **Data Leakage**: Strictly avoid using test set data (October 2022 – April 2023) for training or tuning.
- **Collaboration**: All group members must contribute significantly to technical work and deliverables.
- **Flexibility**: While the Seq2Seq LSTM architecture is specified, variations in implementation are encouraged with proper justification.
- **Tools**: Use open-source frameworks (e.g., TensorFlow, PyTorch, Keras).
- **Citations**: Reference all external packages, papers, or resources used.

## Tentative Grading Scheme
- Technical report correctness and completeness: 40%
- Quality of analysis and interpretation: 30%
- Code quality, reproducibility, and documentation: 20%
- Presentation: 10%

## Timeline
- Model development and preliminary results: 3–4 weeks
- Report and presentation preparation: 1–2 weeks post-results

## Conclusion
This project offers hands-on experience in integrating the National Water Model with DL-based post-processing. You will develop skills in data preprocessing, model design, and forecast evaluation while exploring real-world hydrologic forecasting challenges. We encourage innovative approaches and look forward to your results!

## References
- Han, H., & Morrison, R. R. (2022). Improved runoff forecasting performance through error predictions using a deep-learning approach. *Journal of Hydrology*, 608, 127653.

