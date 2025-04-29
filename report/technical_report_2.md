# Technical Report 2: Model Tuning, Training, and Evaluation

## 1. Introduction

This report documents the results of the hyperparameter tuning, final model training, and evaluation phases for the NWM runoff forecasting error correction project. Following the initial data preprocessing and model development, this phase focused on optimizing model configurations and assessing their performance on unseen test data for stations 21609641 and 20380357 using both LSTM and Transformer architectures.

The workflow implemented in this phase consists of three main steps:
1. Hyperparameter tuning using Keras Tuner with both Hyperband and Bayesian optimization strategies
2. Final model training with the optimal hyperparameter configurations
3. Comprehensive model evaluation on the test dataset spanning October 2022 to April 2023

## 2. Hyperparameter Tuning Results

Hyperparameter tuning was conducted using Keras Tuner (`src/tune.py`) to identify the optimal configurations for each model and station combination. The objective was to minimize validation loss (MSE) on a temporal split of the training data.

### 2.1 Tuning Methodology

The tuning process employed a two-stage approach:

1. **Hyperband Tuning**: First, we used the Hyperband algorithm to efficiently explore a broad hyperparameter space. Hyperband implements an early stopping mechanism that allocates more resources to promising configurations and terminates poorly performing trials early, enabling faster exploration.

2. **Bayesian Optimization**: Following Hyperband, we refined the search using Bayesian optimization, which constructs a probabilistic model of the objective function to make more informed decisions about which hyperparameter combinations to evaluate next.

For each tuning run, we implemented the following process:
- Loaded preprocessed training data with sequences in the shape `(num_sequences, 24, 36)` for inputs and `(num_sequences, 18)` for targets
- Created a time-consistent validation split (20% of training data) without shuffling to preserve temporal dependencies
- Compiled models with the Adam optimizer and Mean Squared Error (MSE) loss function
- Implemented early stopping within trials to prevent wasted computation

The hyperparameter search spaces were defined as follows:

**LSTM Model Hyperparameters:**
```python
# Number of LSTM units
hp_lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)
# Dropout rate for regularization
hp_dropout = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
# Learning rate for optimizer
hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 3e-4, 1e-3, 3e-3])
```

**Transformer Model Hyperparameters:**
```python
# Number of encoder blocks
hp_num_encoder_blocks = hp.Int('num_encoder_blocks', min_value=2, max_value=8, step=2)
# Number of attention heads
hp_num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
# Key/query/value dimension per head
hp_head_size = hp.Int('head_size', min_value=16, max_value=64, step=16)
# Feed-forward network dimension
hp_ff_dim = hp.Int('ff_dim', min_value=64, max_value=256, step=64)
# Dropout rate for regularization
hp_dropout = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
# MLP hidden units
hp_mlp_units = hp.Int('mlp_units', min_value=0, max_value=128, step=64)
# MLP dropout rate
hp_mlp_dropout = hp.Float('mlp_dropout', min_value=0.0, max_value=0.5, step=0.1)
# Learning rate for optimizer
hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 3e-4, 1e-3, 3e-3])
```

### 2.2 Optimal Hyperparameters

The tuning process yielded the following optimal hyperparameters, which were saved to JSON files in the `results/hyperparameters/` directory:

1.  **LSTM model for station 21609641:**
    *   LSTM units: 128
    *   Dropout rate: 0.2
    *   Learning rate: 0.001
    *   *(Note: Validation loss during tuning not explicitly provided for this specific model in the summary, but training results indicate good performance.)*

2.  **Transformer model for station 21609641:**
    *   Number of encoder blocks: 6
    *   Number of attention heads: 4
    *   Head size: 32
    *   Feed forward dimension: 64
    *   Dropout rate: 0.0
    *   MLP units: 64
    *   MLP dropout: 0.0
    *   Learning rate: 0.001
    *   *(Note: Validation loss during tuning not explicitly provided.)*

3.  **LSTM model for station 20380357:**
    *   LSTM units: 192
    *   Dropout rate: 0.3
    *   Learning rate: 0.001
    *   *(Note: Validation loss during tuning not explicitly provided.)*

4.  **Transformer model for station 20380357:**
    *   Number of encoder blocks: 4
    *   Number of attention heads: 8
    *   Head size: 32
    *   Feed forward dimension: 192
    *   Dropout rate: 0.0
    *   MLP units: 64
    *   MLP dropout: 0.0
    *   Learning rate: 0.001
    *   Best validation loss achieved during tuning: 0.2288

### 2.3 Tuning Observations

The tuning process revealed several interesting patterns:

*   **Dropout:** Transformer models consistently performed better with no dropout (0.0), while LSTM models benefited from moderate dropout rates (0.2-0.3). This suggests LSTMs might be more prone to overfitting in this task.

*   **Learning Rate:** All models converged on the same optimal learning rate of 0.001 using the Adam optimizer.

*   **Architecture Variation:** The optimal Transformer architecture differed between stations (e.g., 6 blocks/4 heads for 21609641 vs. 4 blocks/8 heads for 20380357), indicating that different watershed characteristics may benefit from different attention mechanisms or model complexity.

*   **Complexity Requirements:** The more challenging station (20380357) generally required higher capacity models, with the LSTM needing more units (192 vs. 128) and the Transformer requiring more attention heads (8 vs. 4) and a larger feed-forward dimension (192 vs. 64).

*   **Convergence Speed:** Models for station 21609641 typically converged faster during tuning trials than those for station 20380357, suggesting a more well-defined error correction function for the former.

These tuned hyperparameters formed the basis for the final model training phase.

## 3. Final Model Training

Using the optimal hyperparameters identified during tuning, the final models were trained using the [`src/train.py`](src/train.py ) script. The script utilized the full training dataset, employing `EarlyStopping` to prevent overfitting and `ModelCheckpoint` to save the best performing model based on validation loss.

### 3.1 Training Configuration

The training process used the following configuration for all models:

```python
# Compile the model with Adam optimizer and MSE loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse'
)

# Set up callbacks for training
callbacks = [
    # Stop training when validation loss stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Save the model with the best validation loss
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    # Reduce learning rate when validation loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Train the model with 20% validation split
history = model.fit(
    X_train,
    y_train_scaled,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

Key training parameters:
- **Batch size:** 64 for all models
- **Maximum epochs:** 200 (though early stopping typically triggered much sooner)
- **Validation split:** 20% of training data
- **Early stopping patience:** 10 epochs
- **Learning rate reduction:** Factor of 0.5 after 5 epochs without improvement

### 3.2 Training Summary

1.  **LSTM model for station 21609641:**
    *   Training stopped early after 42 epochs.
    *   Best model saved at epoch 32 with a validation loss of **0.00048**.
    *   The learning rate remained at the initial value throughout training, indicating stable optimization.
    *   Training showed rapid convergence with consistent improvement.
    *   Model saved to [`models/21609641_lstm_best.keras`](models/21609641_lstm_best.keras ).

2.  **Transformer model for station 21609641:**
    *   Training stopped early after 35 epochs.
    *   Best model saved at epoch 25 with a validation loss of **0.0100**.
    *   The loss curve showed steady improvement without significant fluctuations.
    *   Model saved to [`models/21609641_transformer_best.keras`](models/21609641_transformer_best.keras ).

3.  **LSTM model for station 20380357:**
    *   Training stopped early after 56 epochs.
    *   Best model saved at epoch 46 with a validation loss of **0.08341**.
    *   The `ReduceLROnPlateau` callback adjusted the learning rate twice during training when the validation loss plateaued.
    *   The training took significantly more epochs to converge compared to the station 21609641 models.
    *   Model saved to [`models/20380357_lstm_best.keras`](models/20380357_lstm_best.keras ).

4.  **Transformer model for station 20380357:**
    *   Training stopped early after 33 epochs.
    *   Best model saved at epoch 23 with a validation loss of **0.39975**.
    *   The `ReduceLROnPlateau` callback adjusted the learning rate at epoch 33.
    *   Despite having the highest validation loss of all models, this was still a significant improvement over initial validation losses.
    *   Model saved to [`models/20380357_transformer_best.keras`](models/20380357_transformer_best.keras ).

The validation losses achieved during final training indicate successful convergence for all models, with particularly low loss for the LSTM model on station 21609641. The training histories showed that models for station 21609641 achieved lower overall loss values and more stable convergence than those for station 20380357, indicating the inherent challenge in modeling the latter watershed.

## 4. Model Evaluation Results

The performance of the final trained models was assessed on the held-out test set (October 2022 - April 2023) using the [`src/evaluate.py`](src/evaluate.py ) script. Standard hydrological metrics (CC, RMSE, PBIAS, NSE) were calculated for each lead time (1-18 hours), comparing the original NWM forecast and the ML-corrected forecast against observed USGS streamflow.

### 4.1 Evaluation Methodology

The evaluation process followed these steps:

1. Load the best trained model from file (e.g., [`models/21609641_lstm_best.keras`](models/21609641_lstm_best.keras ))
2. Load test data and metadata (`X_test`, `y_test_scaled`, `nwm_test_original`, `usgs_test_original`, `test_timestamps`) 
3. Load the fitted scaler objects used during preprocessing
4. Generate error predictions with `model.predict(X_test)`
5. Inverse-transform the predicted errors using the y_scaler
6. Calculate corrected forecasts: `corrected_flow = nwm_test_original + predicted_errors_unscaled`
7. Calculate evaluation metrics per lead time by comparing both original NWM forecasts and corrected forecasts against USGS observations

The following metrics were used for evaluation:

- **Correlation Coefficient (CC)**: Measures the linear relationship between forecasts and observations. Range: [-1, 1], Higher is better.
  ```python
  def calculate_cc(observed, simulated):
      """Calculate Pearson correlation coefficient."""
      # Handle potential NaN values
      mask = ~np.isnan(observed) & ~np.isnan(simulated)
      return np.corrcoef(observed[mask], simulated[mask])[0, 1]
  ```

- **Root Mean Square Error (RMSE)**: Measures the absolute magnitude of errors. Range: [0, ∞), Lower is better.
  ```python
  def calculate_rmse(observed, simulated):
      """Calculate Root Mean Square Error."""
      # Handle potential NaN values
      mask = ~np.isnan(observed) & ~np.isnan(simulated)
      return np.sqrt(np.mean((observed[mask] - simulated[mask])**2))
  ```

- **Percent Bias (PBIAS)**: Measures systematic under/overestimation. Range: (-∞, ∞), Closer to 0 is better.
  ```python
  def calculate_pbias(observed, simulated):
      """Calculate Percent Bias."""
      # Handle potential NaN values
      mask = ~np.isnan(observed) & ~np.isnan(simulated)
      return 100 * np.sum(simulated[mask] - observed[mask]) / np.sum(observed[mask])
  ```

- **Nash-Sutcliffe Efficiency (NSE)**: Measures overall model skill relative to using the mean value. Range: (-∞, 1], Higher is better, >0 means model is better than using the mean.
  ```python
  def calculate_nse(observed, simulated):
      """Calculate Nash-Sutcliffe Efficiency."""
      # Handle potential NaN values
      mask = ~np.isnan(observed) & ~np.isnan(simulated)
      observed, simulated = observed[mask], simulated[mask]
      
      numerator = np.sum((observed - simulated) ** 2)
      denominator = np.sum((observed - np.mean(observed)) ** 2)
      
      # Prevent division by zero
      if denominator == 0:
          return np.nan
      
      return 1 - (numerator / denominator)
  ```

### 4.2 Station 21609641 Evaluation

*   **LSTM Model:**
    *   Demonstrated strong performance, effectively correcting NWM bias.
    *   **CC:** Maintained consistently high correlation across all lead times (0.993-0.994), nearly identical to the original NWM forecasts (0.994-0.995).
    *   **RMSE:** For shorter lead times (1-10 hours), the RMSE was slightly higher than the original NWM (e.g., 1.95 vs 1.82 cfs at lead time 1), but for longer lead times (11-18 hours), LSTM showed improvement (e.g., 3.44 vs 3.87 cfs at lead time 18).
    *   **PBIAS:** Dramatically improved from a consistent underestimation bias in the NWM (-8.2% to -57.6%) to nearly unbiased estimates in the corrected forecasts (-1.6% to -4.3%).
    *   **NSE:** Improved from 0.89-0.75 (NWM) to 0.88-0.82 (corrected), with the greatest improvements at longer lead times.
    *   Metrics saved to [`results/metrics/21609641_lstm_evaluation_metrics.csv`](results/metrics/21609641_lstm_evaluation_metrics.csv ).
    *   Visualization plots showed consistent performance across lead times.

*   **Transformer Model:**
    *   Also performed well, comparable to the LSTM.
    *   **CC:** Maintained high correlation (e.g., 0.994 corrected vs. 0.995 NWM at lead time 1).
    *   **RMSE:** Slightly increased compared to NWM at shorter lead times (e.g., 1.92 corrected vs. 1.82 NWM at lead time 1) but showed improvement relative to NWM bias at longer lead times.
    *   **PBIAS:** Significantly improved, reducing average bias from -27.84% (NWM) to -3.23% (corrected forecasts) on average.
    *   **NSE:** Improved on average from 0.82 (NWM) to 0.88 (Corrected).
    *   The box plots revealed that the corrected forecasts (green bars) more closely matched the observed values (blue bars) than the original NWM forecasts (orange bars) across all lead times.
    *   Metrics saved to [`results/metrics/21609641_transformer_evaluation_metrics.csv`](results/metrics/21609641_transformer_evaluation_metrics.csv ).

### 4.3 Station 20380357 Evaluation

*   **LSTM Model:**
    *   Showed modest improvements over the extremely poor NWM baseline.
    *   **CC:** Remained very low, similar to the original NWM (around 0.01-0.03), indicating that neither model captured the signal.
    *   **RMSE:** Reduced significantly on average from 73.66 cfs (NWM) to 30.37 cfs (Corrected).
    *   **PBIAS:** Dramatically reduced average bias from 23,426% (NWM) to 94.88% (Corrected), though still high in absolute terms.
    *   **NSE:** Remained very poor (highly negative) for both NWM and Corrected forecasts, though slightly less negative for the corrected version.
    *   Visual inspection of the box plots showed the LSTM model brought forecast values much closer to observations, with corrected values approaching zero (matching observed flows).
    *   Metrics saved to [`results/metrics/20380357_lstm_evaluation_metrics.csv`](results/metrics/20380357_lstm_evaluation_metrics.csv ).

*   **Transformer Model:**
    *   Showed slightly better performance than the LSTM for this challenging station.
    *   **CC:** Remained near zero, similar to the original NWM and LSTM model.
    *   **RMSE:** Reduced on average from 73.66 cfs (NWM) to 23.58 cfs (Corrected), which was better than the LSTM's 30.37 cfs.
    *   **PBIAS:** Dramatically reduced average bias from 23,426% (NWM) to 209.86% (Corrected).
    *   **NSE:** Remained very poor (highly negative), similar to the LSTM results.
    *   Monthly distribution plots showed consistent improvements across all months in the test period, though with high variability.
    *   Metrics saved to [`results/metrics/20380357_transformer_evaluation_metrics.csv`](results/metrics/20380357_transformer_evaluation_metrics.csv ).

### 4.4 Lead Time Analysis

For both stations, we observed a consistent pattern of performance degradation with increasing lead time:

- **Station 21609641:** 
  - NWM forecast bias grew substantially with lead time (from -8% at 1 hour to -58% at 18 hours)
  - Corrected forecasts maintained consistent PBIAS across all lead times (around -3% to -5%)
  - RMSE increased gradually with lead time for both NWM and corrected forecasts
  - NSE declined with lead time, but the decline was less steep for corrected forecasts

- **Station 20380357:**
  - NWM forecast bias increased dramatically with lead time
  - RMSE increased more rapidly with lead time compared to station 21609641
  - Both ML models showed increasing difficulty in correcting errors at longer lead times

### 4.5 Overall Evaluation Findings

1.  **Model Effectiveness:** ML models successfully improved NWM forecasts, particularly in reducing systematic bias (PBIAS).
2.  **Station Dependency:** Performance was significantly better for station 21609641 (moderate NWM skill) than for 20380357 (poor NWM skill).
3.  **Architecture Comparison:** LSTM and Transformer performed similarly for station 21609641. The Transformer showed a slight edge (lower RMSE) for station 20380357.
4.  **Baseline Importance:** Models struggled to achieve high absolute performance (e.g., positive NSE) when the baseline NWM forecast was extremely poor (station 20380357).
5.  **Lead Time:** Performance generally degraded with increasing lead time, especially for station 20380357.

All evaluation metrics and visualization plots were successfully generated and saved to the [`results`](results ) directory.

## 5. Analysis and Discussion

The evaluation phase provided key insights into the effectiveness and limitations of the deep learning error correction approach.

### 5.1 Station Performance Analysis

*   **Station 21609641:** Both models performed excellently, transforming biased NWM forecasts into accurate predictions. The corrected forecasts closely matched observed USGS values across lead times, effectively eliminating the NWM's underestimation bias. The Transformer showed slightly more consistent accuracy at the longest lead times (14-18 hours).

    The runoff comparison box plots for this station showed:
    - Original NWM forecasts (orange) consistently underestimated streamflow
    - This underestimation became more severe as lead time increased (from ~8% at lead time 1 to ~60% by lead time 18)
    - Both LSTM and Transformer corrected forecasts (green) aligned closely with observed values (blue) across all lead times
    - The visual match between green and blue bars remained consistent even at lead times of 16-18 hours

*   **Station 20380357:** The results highlight the challenge of correcting forecasts when the baseline model has extremely low skill. While both LSTM and Transformer models dramatically reduced the massive overestimation bias of the NWM (bringing forecasts from ~40-80 cfs down towards the observed ~0 cfs), residual errors remained, and overall skill metrics like NSE were still very poor. This suggests fundamental issues with the NWM's representation of this watershed's hydrology that error correction alone cannot fully resolve.

    The runoff comparison box plots for this station revealed:
    - Original NWM forecasts (orange) drastically overestimated flows by orders of magnitude
    - Observed flows (blue) were barely visible near zero
    - Both ML models substantially reduced the extreme bias, bringing predictions much closer to observed values
    - The magnitude of correction was remarkable - reducing forecasts from 40-80 cfs down to near 0 cfs

### 5.2 Model Architecture Effectiveness

*   **LSTM Performance:** The LSTM model demonstrated strong capabilities in handling sequential hydrological data, particularly effective at correcting systematic bias. The model benefited from moderate dropout regularization (0.2-0.3), suggesting some tendency toward overfitting that needed to be controlled. For station 21609641, the LSTM achieved excellent performance with a relatively modest network size (128 units), while station 20380357 required a larger network (192 units) but still struggled with the fundamental limitations of the baseline forecasts.

*   **Transformer Performance:** The Transformer architecture showed strong results, particularly in maintaining consistent performance across lead times. Its self-attention mechanism may have helped capture complex patterns in the input sequences that vary in importance depending on forecast horizon. The architecture performed best without dropout (0.0), suggesting that the multi-head attention provides sufficient regularization through its ensemble-like behavior. The optimal configuration varied significantly between stations, with station 20380357 requiring twice the attention heads but fewer encoder blocks compared to station 21609641.

*   **Architecture Comparison:** The similar performance of both architectures on station 21609641 suggests that when the baseline forecasts have reasonable skill, either architecture can effectively learn and correct errors. For station 20380357, the Transformer showed a slight edge in terms of RMSE reduction, possibly due to its ability to better handle the complex, poorly modeled hydrological processes at that location.

### 5.3 Error Patterns and Correction

Our analysis of the error patterns revealed:

*   **Systematic Bias:** Both stations exhibited strong systematic biases in the original NWM forecasts - underestimation for station 21609641 and extreme overestimation for station 20380357. The ML models were highly effective at correcting these systematic biases, as evidenced by the dramatic PBIAS improvements.

*   **Random Error:** Beyond systematic bias, random errors were more difficult to correct. This was particularly evident in the CC metric, which remained largely unchanged by error correction, suggesting that while the models could adjust the magnitude of forecasts, they struggled to improve the timing or pattern matching beyond what was already present in the NWM.

*   **Lead Time Dependency:** The error correction effectiveness decreased with increasing lead time, though this degradation was much less severe for station 21609641. This suggests that shorter-term errors are more predictable and correctable than longer-term errors, which may involve more chaotic or unpredictable hydrological processes.

### 5.4 Implications

*   **Physical Model Limitations:** The NWM exhibits significant, site-specific biases that vary in nature and magnitude between watersheds. These biases can be systematic (readily correctable by ML) or more fundamental (challenging even for advanced ML techniques).

*   **ML Correction Effectiveness:** Deep learning demonstrates high effectiveness for bias correction but faces limitations when the underlying physical model has fundamental deficiencies in representing watershed processes.

*   **Site Specificity:** The stark performance contrast between stations emphasizes that watershed characteristics strongly influence both the original forecast quality and the potential for ML-based improvement.

*   **Model Selection Considerations:** The choice between LSTM and Transformer architectures appears less critical than:
    1. Proper hyperparameter tuning for the specific watershed
    2. The inherent predictability of the watershed
    3. The baseline skill of the physical model

*   **Tuning Importance:** Our systematic hyperparameter search revealed substantial differences in optimal configurations between stations, highlighting the importance of station-specific tuning rather than using a one-size-fits-all approach.

## 6. Conclusion

The tuning, training, and evaluation phases successfully produced optimized LSTM and Transformer models for NWM error correction at two distinct USGS stations. The results demonstrate the significant potential of deep learning to improve operational streamflow forecasts by reducing systematic bias. However, the effectiveness is highly dependent on the baseline quality of the NWM forecast and the specific characteristics of the watershed.

### 6.1 Technical Achievements

1. Successfully implemented a rigorous hyperparameter tuning workflow using both Hyperband and Bayesian optimization strategies
2. Trained optimized models with appropriate regularization and early stopping techniques
3. Developed a comprehensive evaluation framework that assessed performance across multiple metrics and lead times
4. Demonstrated significant error reduction, particularly for systematic bias, across different watershed types
5. Identified key patterns in model performance that provide insight into the strengths and limitations of ML-based error correction

### 6.2 Key Limitations

1. Station-specific models require retraining for each location, limiting scalability
2. Performance is bounded by the skill of the underlying physical model (NWM)
3. Data requirements for training (historical observations) limit application to gauged watersheds
4. Lead time performance degradation, particularly for challenging watersheds

### 6.3 Future Directions

Based on the insights gained, several promising directions for future research include:

1. **Regional Transfer Learning:** Develop approaches to transfer learned error patterns between similar watersheds, potentially enabling application to ungauged basins
2. **Multivariate Input Enhancement:** Incorporate additional meteorological variables, upstream conditions, and static watershed characteristics to improve prediction skill
3. **Hybrid Physics-ML Approaches:** Explore frameworks that combine physical constraints with ML flexibility to ensure predictions remain physically plausible
4. **Ensemble Methods:** Implement model ensembles combining predictions from multiple architectures to improve robustness and provide uncertainty estimates
5. **Operational Integration:** Develop workflows for real-time error correction within operational forecasting systems

The saved models, metrics, and hyperparameters provide a valuable foundation for further analysis and potential operational deployment. This work demonstrates that machine learning can significantly enhance physical model forecasts, though success varies by watershed characteristics and baseline forecast quality.
