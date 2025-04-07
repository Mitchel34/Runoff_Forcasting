# Runoff Forecasting with Deep Learning

## Project Overview
This project aims to improve the National Water Model (NWM) forecasts using deep learning techniques. By combining NWM forecasts with USGS observational data, we train models to correct systematic biases and improve runoff predictions across various watersheds.

## Directory Structure
```
nwm_dl_postprocessing/
├── data/
│   ├── raw/                           # Original data sources
│   │   ├── nwm_forecasts.csv          # NWM forecasts (hourly, 1-18h lead)
│   │   └── usgs_observations.csv      # USGS observed runoff data
│   └── processed/                     # Cleaned and prepared datasets
│       ├── train_validation_data.csv  # Data for training/validation (Apr 2021-Sep 2022)
│       └── test_data.csv              # Data for testing (Oct 2022-Apr 2023)
├── models/
│   └── nwm_dl_model.keras             # Saved model file
├── notebooks/
│   ├── exploratory_analysis.ipynb     # Data exploration notebook
│   ├── model_development.ipynb        # Model building notebook
│   └── results_visualization.ipynb    # Results and visualization notebook
├── src/
│   ├── preprocess.py                  # Data preprocessing scripts
│   ├── model.py                       # Model definition and training
│   ├── predict.py                     # Prediction script
│   ├── evaluate.py                    # Model evaluation metrics
│   └── visualize.py                   # Visualization utilities
├── tests/
│   ├── test_preprocess.py             # Tests for preprocessing
│   └── test_model.py                  # Tests for model functionality
├── reports/
│   ├── figures/                       # Generated figures
│   │   ├── runoff_boxplots.png        # Comparison boxplots
│   │   └── metrics_boxplots.png       # Performance metrics
│   └── technical_report.pdf           # Technical documentation
├── presentation/
│   ├── presentation_slides.pdf        # Presentation slides
│   └── presentation_notes.md          # Speaker notes
├── requirements.txt                   # Project dependencies
├── README.md                          # This file
└── .gitignore                         # Git exclusions
```

## Installation & Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Runoff_Forcasting
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the data files (large files are not stored in the repository):
   ```
   python scripts/download_data.py
   ```
   
   Alternatively, you can manually download the data files from [our data storage](https://link-to-your-data-storage) and place them in the appropriate directories.

## Data

- **Training/Validation Data**: April 2021 to September 2022
- **Testing Data**: October 2022 to April 2023

The data includes:
- NWM forecasts with hourly predictions at 1-18 hour lead times
- USGS observational runoff data for ground truth

**Note**: Large data files are not stored directly in the repository. Use the download script to retrieve them.

## Model

We implement several deep learning architectures:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer-based models
- Hybrid approaches

See `src/model.py` for implementation details.

## Evaluation Metrics

The models are evaluated using standard hydrological metrics:
- Correlation Coefficient (CC)
- Root Mean Square Error (RMSE)
- Percent Bias (PBIAS)
- Nash-Sutcliffe Efficiency (NSE)

## Usage

### Data Preprocessing
```
python src/preprocess.py
```

### Model Training
```
python src/model.py --train
```

### Making Predictions
```
python src/predict.py --input <input_file> --output <output_file>
```

### Evaluation
```
python src/evaluate.py --predictions <predictions_file> --observations <observations_file>
```

### Visualization
```
python src/visualize.py --results <results_file>
```

## Notebooks

- `exploratory_analysis.ipynb`: Initial data exploration and insights
- `model_development.ipynb`: Interactive model development and tuning
- `results_visualization.ipynb`: Final results and visualizations

## Results

The project demonstrates improved runoff forecasting compared to raw NWM predictions, with detailed performance metrics available in the `reports/` directory.

## Contributors

- Your Name
- Team Members

## License

[Specify license information]
