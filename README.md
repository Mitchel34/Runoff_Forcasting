# Runoff Forecasting: NWM Error Correction with Deep Learning

This project implements deep learning models to improve National Water Model (NWM) runoff forecasts through error prediction. It uses specialized architectures for two distinct USGS stations:
- **Station 21609641**: LSTM-based and Transformer-based error prediction models
- **Station 20380357**: LSTM-based and Transformer-based error prediction models

## Project Setup

### 1. Environment Setup

This project requires Python 3.10 specifically for compatibility with TensorFlow and other dependencies.

```bash
# Clone the repository
git clone <repository-url>
cd runoff_forecasting

# Create and activate a Python 3.10 virtual environment
python3.10 -m venv .venv

# On macOS/Linux
source .venv/bin/activate 

# On Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

The project requires NWM forecast data and USGS observation data for two stations:
- Station 21609641 (USGS gauge 11266500)
- Station 20380357 (USGS gauge 09520500)

Ensure your data is organized in the `data/raw/<station_id>/` directories.

### 3. Running the Complete Workflow

#### A. Data Preprocessing

Preprocess data for both stations (creates training/testing datasets and scales data):

```bash
python src/preprocess.py
```

This creates sequences with a 24-hour lookback window to predict 18 hours of forecast errors.

#### B. Hyperparameter Tuning (for all 4 models)

For best results, we use a two-stage tuning approach (Hyperband followed by Bayesian Optimization):

##### Stage 1: Hyperband Tuning

```bash
# 1. LSTM for Station 21609641
python src/tune.py --station_id 21609641 --model_type lstm --tuner hyperband --epochs_per_trial 50 --batch_size 64 --max_trials 30

# 2. Transformer for Station 21609641 
python src/tune.py --station_id 21609641 --model_type transformer --tuner hyperband --epochs_per_trial 50 --batch_size 64 --max_trials 30

# 3. LSTM for Station 20380357
python src/tune.py --station_id 20380357 --model_type lstm --tuner hyperband --epochs_per_trial 50 --batch_size 64 --max_trials 30

# 4. Transformer for Station 20380357
python src/tune.py --station_id 20380357 --model_type transformer --tuner hyperband --epochs_per_trial 50 --batch_size 64 --max_trials 30
```

##### Stage 2: Bayesian Optimization (refining the search)

```bash
# 1. LSTM for Station 21609641
python src/tune.py --station_id 21609641 --model_type lstm --tuner bayesian --epochs_per_trial 80 --batch_size 64 --max_trials 20

# 2. Transformer for Station 21609641 
python src/tune.py --station_id 21609641 --model_type transformer --tuner bayesian --epochs_per_trial 80 --batch_size 64 --max_trials 20

# 3. LSTM for Station 20380357
python src/tune.py --station_id 20380357 --model_type lstm --tuner bayesian --epochs_per_trial 80 --batch_size 64 --max_trials 20

# 4. Transformer for Station 20380357
python src/tune.py --station_id 20380357 --model_type transformer --tuner bayesian --epochs_per_trial 80 --batch_size 64 --max_trials 20
```

The best hyperparameters will be saved to `.json` files in the `results/hyperparameters` directory. Review the console output or these files to find the optimal configurations.

#### C. Model Training (with optimal hyperparameters)

Train the four models using the hyperparameters found during tuning:

```bash
# 1. LSTM for Station 21609641
python src/train.py --station_id 21609641 --model_type lstm --epochs 100 --batch_size 64 \
    --lstm_units 128 --dropout_rate 0.2 --lr 0.001

# 2. Transformer for Station 21609641
python src/train.py --station_id 21609641 --model_type transformer --epochs 100 --batch_size 64 \
    --num_encoder_blocks 6 --num_heads 4 --head_size 32 --ff_dim 64 --dropout_rate 0.0 \
    --mlp_units 64 --mlp_dropout 0.0 --lr 0.001

# 3. LSTM for Station 20380357
python src/train.py --station_id 20380357 --model_type lstm --epochs 100 --batch_size 64 \
    --lstm_units 192 --dropout_rate 0.3 --lr 0.001

# 4. Transformer for Station 20380357
python src/train.py --station_id 20380357 --model_type transformer --epochs 100 --batch_size 64 \
    --num_encoder_blocks 4 --num_heads 8 --head_size 32 --ff_dim 192 --dropout_rate 0.0 \
    --mlp_units 64 --mlp_dropout 0.0 --lr 0.001
```

**Note**: You can also load hyperparameters from the saved JSON files:

```bash
# Example of using saved hyperparameters
python src/train.py --station_id 21609641 --model_type lstm --epochs 100 --batch_size 64 \
    --hyperparams_file results/hyperparameters/21609641_lstm_best_hps.json
```

#### D. Model Evaluation

Evaluate each model on the test dataset (Oct 2022 - Apr 2023):

```bash
# 1. LSTM for Station 21609641
python src/evaluate.py --station_id 21609641 --model_type lstm

# 2. Transformer for Station 21609641
python src/evaluate.py --station_id 21609641 --model_type transformer

# 3. LSTM for Station 20380357
python src/evaluate.py --station_id 20380357 --model_type lstm

# 4. Transformer for Station 20380357
python src/evaluate.py --station_id 20380357 --model_type transformer
```

This generates evaluation metrics and visualizations in the `results` directory:
- Metrics are saved as CSV files in `results/metrics`
- Plots are saved as PNG files in `results/plots`

### 4. Expected Results

The evaluation generates multiple metrics for comparing original NWM forecasts with corrected forecasts:
- **Correlation Coefficient (CC)**: Measures timing/pattern match
- **Root Mean Square Error (RMSE)**: Measures absolute magnitude of errors
- **Percent Bias (PBIAS)**: Measures systematic under/overestimation
- **Nash-Sutcliffe Efficiency (NSE)**: Measures overall model skill

Based on prior runs, expect:
- **Station 21609641**: Both models significantly improve PBIAS and NSE
- **Station 20380357**: Both models struggle due to poor baseline NWM data, but still reduce RMSE and PBIAS

## Visualizing Results

After running the evaluation, review the generated plots in `results/plots`:
- Box plots showing flow distributions across lead times
- Line plots showing metrics vs. lead times
- Monthly distribution plots for key metrics

## Authors

Mitchel Carson & Christian Castaneda
