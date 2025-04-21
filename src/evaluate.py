import os
import sys
import glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Add the project root to the Python path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import correlation_coefficient, rmse, pbias, nse


def load_dataset(npz_path):
    data = np.load(npz_path)
    X, y = data['X'], data['y']
    return X[..., np.newaxis], y


def evaluate_station(station, model_path, test_path, out_plots, out_metrics, suffix=''):
    # load data and model
    X_test, y_true = load_dataset(test_path)
    model = load_model(model_path)
    # extract seq lengths
    window = X_test.shape[1] - y_true.shape[1]
    horizon = y_true.shape[1]
    # forecasts and true obs
    frc_seq = X_test[:, window:, 0]
    obs_true = frc_seq + y_true
    # predictions
    y_pred = model.predict(X_test)
    corrected = frc_seq + y_pred
    # metrics
    metrics = {'lead_time': [], 'CC_raw': [], 'CC_corr': [], 'RMSE_raw': [], 'RMSE_corr': [], 'PBIAS_raw': [], 'PBIAS_corr': [], 'NSE_raw': [], 'NSE_corr': []}
    for h in range(horizon):
        metrics['lead_time'].append(h+1)
        o = obs_true[:, h]
        f = frc_seq[:, h]
        c = corrected[:, h]
        metrics['CC_raw'].append(correlation_coefficient(o, f))
        metrics['CC_corr'].append(correlation_coefficient(o, c))
        metrics['RMSE_raw'].append(rmse(o, f))
        metrics['RMSE_corr'].append(rmse(o, c))
        metrics['PBIAS_raw'].append(pbias(o, f))
        metrics['PBIAS_corr'].append(pbias(o, c))
        metrics['NSE_raw'].append(nse(o, f))
        metrics['NSE_corr'].append(nse(o, c))
    # save metrics table
    dfm = pd.DataFrame(metrics)
    os.makedirs(out_metrics, exist_ok=True)
    csv_path = os.path.join(out_metrics, f'{station}{suffix}_metrics.csv')
    dfm.to_csv(csv_path, index=False)
    # box-plots per lead time
    os.makedirs(out_plots, exist_ok=True)
    for h in range(horizon):
        fig, ax = plt.subplots()
        ax.boxplot([obs_true[:, h], frc_seq[:, h], corrected[:, h]], labels=['Obs', 'NWM', 'Corrected'])
        ax.set_title(f'Station {station}{suffix} Lead Time {h+1}')
        plt.savefig(os.path.join(out_plots, f'{station}{suffix}_box_lead{h+1}.png'))
        plt.close(fig)
    # metric line plots
    metrics_list = ['CC', 'RMSE', 'PBIAS', 'NSE']
    for m in metrics_list:
        fig, ax = plt.subplots()
        ax.plot(metrics['lead_time'], metrics[f'{m}_raw'], label='NWM')
        ax.plot(metrics['lead_time'], metrics[f'{m}_corr'], label='Corrected')
        ax.set_xlabel('Lead Time')
        ax.set_ylabel(m)
        ax.set_title(f'Station {station}{suffix} {m}')
        ax.legend()
        plt.savefig(os.path.join(out_plots, f'{station}{suffix}_{m}.png'))
        plt.close(fig)
    print(f'Evaluated station {station}{suffix}. Metrics saved to {csv_path}. Plots saved to {out_plots}')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', default='models', help='Directory containing saved models')
    parser.add_argument('--data-dir', default=os.path.join('data', 'processed'), help='Processed data directory')
    parser.add_argument('--plots-dir', default=os.path.join('results', 'plots'), help='Directory to save plots')
    parser.add_argument('--metrics-dir', default=os.path.join('results', 'metrics'), help='Directory to save metric CSVs')
    parser.add_argument('--use-tuned', action='store_true', help='Evaluate tuned models (with _tuned suffix)')
    parser.add_argument('--compare', action='store_true', help='Compare default and tuned models')
    args = parser.parse_args()

    test_files = glob.glob(os.path.join(args.data_dir, 'test', '*.npz'))
    for tp in test_files:
        station = os.path.splitext(os.path.basename(tp))[0]

        # Evaluate default model
        if not args.use_tuned or args.compare:
            model_path = os.path.join(args.models_dir, f'{station}.h5')
            if os.path.exists(model_path):
                metrics_default = evaluate_station(station, model_path, tp, args.plots_dir, args.metrics_dir)
            else:
                print(f"Default model not found for station {station}")
                metrics_default = None

        # Evaluate tuned model
        if args.use_tuned or args.compare:
            tuned_model_path = os.path.join(args.models_dir, f'{station}_tuned.h5')
            if os.path.exists(tuned_model_path):
                metrics_tuned = evaluate_station(station, tuned_model_path, tp, args.plots_dir, args.metrics_dir, suffix='_tuned')
            else:
                print(f"Tuned model not found for station {station}")
                metrics_tuned = None

        # Generate comparison report if both models were evaluated
        if args.compare and metrics_default and metrics_tuned:
            # Create comparison plots
            metrics_list = ['CC', 'RMSE', 'PBIAS', 'NSE']
            for m in metrics_list:
                fig, ax = plt.subplots()
                ax.plot(metrics_default['lead_time'], metrics_default[f'{m}_corr'], label='Default Model')
                ax.plot(metrics_tuned['lead_time'], metrics_tuned[f'{m}_corr'], label='Tuned Model')
                ax.plot(metrics_default['lead_time'], metrics_default[f'{m}_raw'], label='Raw NWM', linestyle='--')
                ax.set_xlabel('Lead Time')
                ax.set_ylabel(m)
                ax.set_title(f'Station {station} {m} Comparison')
                ax.legend()
                plt.savefig(os.path.join(args.plots_dir, f'{station}_compare_{m}.png'))
                plt.close(fig)

            print(f'Generated comparison plots for station {station}')


if __name__ == '__main__':
    main()