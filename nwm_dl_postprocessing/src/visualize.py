import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class ForecastVisualizer:
    """
    Creates visualizations for forecast comparison and evaluation metrics.
    Generates box plots of runoff values and evaluation metrics across lead times.
    """
    
    def __init__(self, figures_path="../reports/figures"):
        """
        Initialize the ForecastVisualizer.
        
        Parameters:
        -----------
        figures_path : str
            Path to save generated figures
        """
        self.figures_path = figures_path
        
        # Make sure figures directory exists
        os.makedirs(figures_path, exist_ok=True)
        
        # Set default style
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})
    
    def create_runoff_boxplots(self, results_df, lead_times=None, save_fig=True):
        """
        Create box plots comparing observed, NWM, LSTM-corrected, and baseline-corrected runoff
        for each lead time.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame containing observed values and various forecasts
        lead_times : list, optional
            Specific lead times to plot (default: 1, 6, 12, 18)
        save_fig : bool
            Whether to save the figure to disk
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if lead_times is None:
            lead_times = [1, 6, 12, 18]  # Default representative lead times
        
        # Make sure lead_times is a list to support iteration
        if not isinstance(lead_times, (list, tuple)):
            lead_times = [lead_times]
        
        # Create figure
        fig, axes = plt.subplots(1, len(lead_times), figsize=(5*len(lead_times), 8), sharey=True)
        
        # If only one lead time, axes is not a list
        if len(lead_times) == 1:
            axes = [axes]
        
        for i, lead in enumerate(lead_times):
            data_to_plot = []
            labels = []
            
            # Get observed runoff
            observed = results_df['usgs_observed'].dropna().values
            data_to_plot.append(observed)
            labels.append('Observed (USGS)')
            
            # Get NWM forecasts
            if f'nwm_lead_{lead}' in results_df.columns:
                nwm_forecasts = results_df[f'nwm_lead_{lead}'].dropna().values
                data_to_plot.append(nwm_forecasts)
                labels.append(f'NWM Forecast ({lead}h)')
            
            # Get LSTM corrected forecasts
            if f'lstm_corrected_lead_{lead}' in results_df.columns:
                lstm_forecasts = results_df[f'lstm_corrected_lead_{lead}'].dropna().values
                data_to_plot.append(lstm_forecasts)
                labels.append(f'LSTM Corrected ({lead}h)')
            
            # Get baseline corrected forecasts
            if f'baseline_corrected_lead_{lead}' in results_df.columns:
                baseline_forecasts = results_df[f'baseline_corrected_lead_{lead}'].dropna().values
                data_to_plot.append(baseline_forecasts)
                labels.append(f'Baseline Corrected ({lead}h)')
            
            # Create box plot
            box = axes[i].boxplot(data_to_plot, patch_artist=True, showfliers=False)
            
            # Color boxes
            colors = ['lightgray', 'lightblue', 'lightgreen', 'salmon']
            for j, patch in enumerate(box['boxes']):
                patch.set_facecolor(colors[j % len(colors)])
            
            # Set labels
            axes[i].set_xticklabels(labels, rotation=45, ha='right')
            axes[i].set_title(f'Lead Time: {lead} hours')
        
        # Set common labels
        fig.text(0.5, 0.01, 'Forecast Type', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Runoff', va='center', rotation='vertical', fontsize=14)
        fig.suptitle('Observed vs NWM vs Corrected (LSTM) vs Corrected (Baseline) Runoff', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.2)
        
        if save_fig:
            plt.savefig(os.path.join(self.figures_path, 'runoff_boxplots.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_metrics_boxplots(self, evaluation_df, save_fig=True):
        """
        Create box plots of evaluation metrics (CC, RMSE, PBIAS, NSE) across lead times.
        
        Parameters:
        -----------
        evaluation_df : pandas.DataFrame
            DataFrame with evaluation metrics
        save_fig : bool
            Whether to save the figure to disk
            
        Returns:
        --------
        figs : list of matplotlib.figure.Figure
            List of created figures
        """
        metrics = ['CC', 'RMSE', 'PBIAS', 'NSE']
        figs = []
        
        # Get forecast types from index
        forecast_types = evaluation_df.index.get_level_values('Forecast').unique()
        
        # Create a plot for each metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data for box plots by lead time
            data_by_lead = []
            lead_times = evaluation_df.index.get_level_values('Lead Time').unique()
            
            # Stack data for each forecast type per lead time
            for lead in sorted(lead_times):
                lead_data = []
                for forecast in forecast_types:
                    try:
                        value = evaluation_df.loc[(forecast, lead), metric]
                        lead_data.append(value)
                    except:
                        lead_data.append(np.nan)
                data_by_lead.append(lead_data)
            
            # Positions for box groups
            positions = np.arange(1, len(lead_times) + 1)
            width = 0.25
            
            # Plot boxes for each forecast type
            for i, forecast in enumerate(forecast_types):
                # Extract values for this forecast type
                forecast_values = [data[i] for data in data_by_lead]
                
                # Position offset for each group
                pos = positions + (i - 1) * width
                
                # Create box plot
                box = ax.boxplot(forecast_values, positions=pos, patch_artist=True, 
                                 showfliers=False, widths=width*0.8)
                
                # Set box color
                colors = ['lightblue', 'lightgreen', 'salmon']
                for patch in box['boxes']:
                    patch.set_facecolor(colors[i % len(colors)])
            
            # Set x-axis labels
            ax.set_xticks(positions)
            ax.set_xticklabels([f'{lead}h' for lead in sorted(lead_times)])
            
            # Set labels
            ax.set_xlabel('Lead Time (hours)', fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.set_title(f'{metric} across Lead Times', fontsize=16)
            
            # Add legend
            ax.legend([plt.Rectangle((0,0), 1, 1, fc=colors[i % len(colors)]) for i in range(len(forecast_types))], 
                     forecast_types, loc='best')
            
            plt.tight_layout()
            
            if save_fig:
                plt.savefig(os.path.join(self.figures_path, f'{metric.lower()}_boxplots.png'), dpi=300, bbox_inches='tight')
            
            figs.append(fig)
        
        # Create a combined figure with all metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            # Prepare data for box plots
            data_for_metric = {}
            for forecast in forecast_types:
                data_for_metric[forecast] = []
                for lead in sorted(lead_times):
                    try:
                        value = evaluation_df.loc[(forecast, lead), metric]
                        data_for_metric[forecast].append(value)
                    except:
                        data_for_metric[forecast].append(np.nan)
            
            # Create DataFrame for seaborn
            plot_data = []
            for forecast, values in data_for_metric.items():
                for j, lead in enumerate(sorted(lead_times)):
                    plot_data.append({
                        'Forecast': forecast,
                        'Lead Time': lead,
                        metric: values[j]
                    })
            plot_df = pd.DataFrame(plot_data)
            
            # Create box plot with seaborn
            sns.boxplot(x='Lead Time', y=metric, hue='Forecast', data=plot_df, ax=axes[i])
            axes[i].set_title(metric)
            
            # Only show legend in first subplot
            if i > 0:
                axes[i].get_legend().remove()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.figures_path, 'metrics_boxplots.png'), dpi=300, bbox_inches='tight')
        
        figs.append(fig)
        
        return figs

if __name__ == "__main__":
    # Example usage
    from preprocess import DataPreprocessor
    from predict import ForecastPredictor
    from baseline import PersistenceBaseline
    from evaluate import ForecastEvaluator
    
    # Load preprocessed data
    preprocessor = DataPreprocessor(
        raw_data_path="../data/raw",
        processed_data_path="../data/processed",
        sequence_length=24
    )
    
    data = preprocessor.process_data(stream_ids=["20380357"])
    
    # Get test data
    test_data = data["20380357"]["test"]
    test_df = test_data["df"]
    
    # Generate LSTM predictions
    predictor = ForecastPredictor("../models/nwm_lstm_model.keras")
    predictor.set_scaler(data["20380357"]["scalers"]["target"])
    lstm_corrected_df = predictor.generate_corrected_forecasts(test_data)
    
    # Generate baseline predictions
    baseline = PersistenceBaseline()
    baseline.train(data["20380357"]["train_val"]["df"])
    baseline_corrected_df = baseline.predict_batch(test_df)
    
    # Combine results
    results_df = lstm_corrected_df.copy()
    for lead in range(1, 19):
        col = f'baseline_corrected_lead_{lead}'
        if col in baseline_corrected_df.columns:
            results_df[col] = baseline_corrected_df[col]
    
    # Evaluate forecasts
    evaluator = ForecastEvaluator()
    evaluation_df = evaluator.evaluate_forecasts(results_df)
    
    # Create visualizations
    visualizer = ForecastVisualizer()
    visualizer.create_runoff_boxplots(results_df)
    visualizer.create_metrics_boxplots(evaluation_df)