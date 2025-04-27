# Runoff Forecasting: NWM Error Correction with Deep Learning

This project implements deep learning models to improve National Water Model (NWM) runoff forecasts through error prediction. It uses specialized architectures for two distinct USGS stations:
- **Station 21609641**: LSTM-based error prediction model
- **Station 20380357**: Transformer-based error prediction model

## Project Structure

```
runoff_forecasting/
├── data/
│   ├── raw/                  # Original NWM and USGS data
│   └── processed/            # Preprocessed data ready for training
├── src/
│   ├── preprocess.py         # Data preprocessing pipeline
│   ├── models/
│   │   ├── lstm.py           # LSTM implementation for station 21609641
│   │   └── transformer.py    # Transformer implementation for station 20380357
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Evaluation and visualization
│   └── utils.py              # Utility functions and metrics
├── notebooks/                # Exploratory data analysis
├── results/                  # Saved results and visualizations
│   ├── plots/                # Generated visualizations
│   └── metrics/              # Performance metrics
├── models/                   # Saved model files
├── report/                   # Technical report
└── requirements.txt          # Project dependencies
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The project requires NWM forecast data and USGS observation data for two stations:
- Station 21609641 (USGS gauge 11266500)
- Station 20380357 (USGS gauge 09520500)


## Workflow

1. **Preprocess data**:
```bash
python src/preprocess.py
```

2. **Train models**:
```bash
python src/train.py --station 21609641  # LSTM for station 21609641
python src/train.py --station 20380357  # Transformer for station 20380357
```

3. **Evaluate and visualize results**:
```bash
python src/evaluate.py
```

## Evaluation Metrics

The models are evaluated using the following hydrological metrics:
- Correlation Coefficient (CC)
- Root Mean Square Error (RMSE)
- Percent Bias (PBIAS)
- Nash-Sutcliffe Efficiency (NSE)

## Authors

Your Name

## License

[MIT](LICENSE)

## Acknowledgments

This project is based on methods from:
- Han, H. & Morrison, R. R. (2022). Improved runoff forecasting performance through error predictions using a deep-learning approach. Journal of Hydrology, 608, 127653.
