import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyArrow, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def create_preprocessing_diagram():
    """Create a comprehensive diagram of the preprocessing pipeline."""
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1.2], height_ratios=[1, 1], 
                 wspace=0.3, hspace=0.3)
    
    # Section 1: Raw Data Format (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    create_raw_data_view(ax1)
    
    # Section 2: Merged & Error Calculation (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    create_merged_data_view(ax2)
    
    # Section 3: Pivoted Data (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    create_pivoted_data_view(ax3)
    
    # Section 4: Sequence Creation (Bottom Span)
    ax4 = fig.add_subplot(gs[1, :])
    create_sequence_diagram(ax4)
    
    # Add title
    plt.suptitle('Data Preprocessing Pipeline: From Raw Data to Model-Ready Sequences', 
                fontsize=18, y=0.98)
    
    # Save the figure
    plt.savefig('preprocessing_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    print("Diagram saved as 'preprocessing_pipeline_diagram.png'")
    plt.show()

def create_raw_data_view(ax):
    """Create view of raw NWM and USGS data."""
    # NWM data sample
    nwm_data = pd.DataFrame({
        'init_time': ['2021-04-20 07:00', '2021-04-20 07:00', '2021-04-20 08:00', '2021-04-20 08:00'],
        'valid_time': ['2021-04-20 08:00', '2021-04-20 09:00', '2021-04-20 09:00', '2021-04-20 10:00'],
        'nwm_flow': [33.7, 35.1, 35.1, 36.0],
        'lead_time': [1, 2, 1, 2]
    })
    
    # USGS data sample
    usgs_data = pd.DataFrame({
        'datetime': ['2021-04-20 08:00', '2021-04-20 09:00', '2021-04-20 10:00'],
        'usgs_flow': [34.0, 35.4, 36.2]
    })
    
    # Display NWM and USGS data tables
    ax.axis('tight')
    ax.axis('off')
    
    # Add NWM table
    nwm_table_text = "NWM Forecast Data:\n\n"
    nwm_table_text += "init_time           valid_time          nwm_flow  lead_time\n"
    nwm_table_text += "2021-04-20 07:00  2021-04-20 08:00    33.7        1\n"
    nwm_table_text += "2021-04-20 07:00  2021-04-20 09:00    35.1        2\n"
    nwm_table_text += "2021-04-20 08:00  2021-04-20 09:00    35.1        1\n"
    nwm_table_text += "2021-04-20 08:00  2021-04-20 10:00    36.0        2"
    
    # Add USGS table
    usgs_table_text = "\n\nUSGS Observation Data:\n\n"
    usgs_table_text += "datetime           usgs_flow\n"
    usgs_table_text += "2021-04-20 08:00    34.0\n"
    usgs_table_text += "2021-04-20 09:00    35.4\n"
    usgs_table_text += "2021-04-20 10:00    36.2"
    
    # Combine text
    table_text = nwm_table_text + usgs_table_text
    
    ax.text(0.5, 0.5, table_text, ha='center', va='center', family='monospace')
    ax.set_title('1. Raw Input Data', fontweight='bold')

def create_merged_data_view(ax):
    """Create view of merged data and error calculation."""
    ax.axis('tight')
    ax.axis('off')
    
    # Add merged table
    merged_table_text = "Merged Data (inner join on valid_time/datetime):\n\n"
    merged_table_text += "datetime           nwm_flow  usgs_flow  lead_time  error\n"
    merged_table_text += "2021-04-20 08:00    33.7      34.0       1        0.3\n"
    merged_table_text += "2021-04-20 09:00    35.1      35.4       1        0.3\n"
    merged_table_text += "2021-04-20 09:00    35.1      35.4       2        0.3\n"
    merged_table_text += "2021-04-20 10:00    36.0      36.2       2        0.2\n"
    
    # Add formula box
    formula_text = "\nError Calculation:\n"
    formula_text += "error = usgs_flow - nwm_flow"
    
    # Combine text
    text = merged_table_text + formula_text
    
    ax.text(0.5, 0.5, text, ha='center', va='center', family='monospace')
    ax.set_title('2. Merged Data & Error Calculation', fontweight='bold')

def create_pivoted_data_view(ax):
    """Create view of pivoted data with lead times as columns."""
    ax.axis('tight')
    ax.axis('off')
    
    # Add pivoted table
    pivot_table_text = "Pivoted Data (lead times as columns):\n\n"
    pivot_table_text += "datetime           nwm_flow_1  nwm_flow_2  usgs_flow_1  usgs_flow_2  error_1  error_2\n"
    pivot_table_text += "2021-04-20 08:00    33.7         -          34.0          -           0.3      -\n"
    pivot_table_text += "2021-04-20 09:00    35.1        35.1        35.4         35.4         0.3      0.3\n"
    pivot_table_text += "2021-04-20 10:00     -          36.0         -           36.2          -       0.2\n"
    
    ax.text(0.5, 0.5, pivot_table_text, ha='center', va='center', family='monospace')
    ax.set_title('3. Pivoted Data (columns for each lead time)', fontweight='bold')

def create_sequence_diagram(ax):
    """Create diagram showing sequence creation with sliding window."""
    ax.axis('off')
    
    # Create sample time series data
    dates = pd.date_range(start='2021-04-20', periods=48, freq='H')
    
    # Sample data for visualization
    np.random.seed(42)
    nwm_flow = 35 + np.cumsum(np.random.normal(0, 0.5, 48))
    usgs_flow = nwm_flow + np.random.normal(0, 1, 48)
    error = usgs_flow - nwm_flow
    
    # Creating dummy data frame to represent our sequence source
    dummy_data = pd.DataFrame({
        'datetime': dates,
        'nwm_flow_1': nwm_flow,
        'usgs_flow_1': usgs_flow,
        'error_1': error
    })
    
    # For clarity, we'll visualize just one feature (error) in the sequence diagram
    time_points = np.arange(48)
    feature_values = error
    
    # Draw time series
    ax.plot(time_points, feature_values, 'b-', alpha=0.5, label='Error Values')
    
    # Define window sizes
    window_size = 24  # Input window (lookback)
    horizon = 18      # Target window (forecast horizon)
    
    # Draw a few sliding windows
    window_starts = [0, 12, 24]  # Starting points for windows
    
    colors = ['lightblue', 'lightgreen', 'salmon']
    for i, start in enumerate(window_starts):
        if start + window_size < len(time_points):
            # Draw input window
            ax.fill_between(
                time_points[start:start+window_size], 
                0, feature_values[start:start+window_size],
                alpha=0.3, color=colors[i], label=f'Input Window {i+1}' if i == 0 else ""
            )
            
            # Draw target point (using a different shape/color)
            if start + window_size < len(time_points):
                end_point = start + window_size
                if end_point + horizon <= len(time_points):
                    # Draw target window
                    ax.fill_between(
                        time_points[end_point:end_point+horizon],
                        0, feature_values[end_point:end_point+horizon],
                        alpha=0.3, color='yellow', label=f'Target Window {i+1}' if i == 0 else ""
                    )
    
    # Annotate sequence structure
    ax.annotate('Input X: 24hr lookback window\n(nwm_flow_1..18, error_1..18)', 
                xy=(window_starts[0] + window_size/2, max(feature_values)*0.8),
                xytext=(window_starts[0] + window_size/2, max(feature_values)*1.2),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center')
    
    ax.annotate('Target y: Next 18 error values', 
                xy=(window_starts[0] + window_size + horizon/2, max(feature_values)*0.8),
                xytext=(window_starts[0] + window_size + horizon/2, max(feature_values)*1.2),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center')
    
    # Additional annotations
    ax.text(0.5, -0.1, 
           "Sliding Window Creation: Each input sequence (24 hours) maps to target errors (18 lead times)\n"
           "Train-Test Split: Apr 2021 - Sep 2022 (Train), Oct 2022 - Apr 2023 (Test)\n"
           "Scaling: StandardScaler fitted on training data only", 
           ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    # Add legend and title
    ax.legend(loc='upper right')
    ax.set_title('4. Sequence Creation - Sliding Window Approach', fontweight='bold')
    
    # Add x and y labels
    ax.set_xlabel('Time Steps (Hours)')
    ax.set_ylabel('Feature Value (Error)')

if __name__ == "__main__":
    create_preprocessing_diagram()