import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Set style with minimal overhead
plt.style.use('default')

def create_preprocessing_diagram_light():
    """Create a simpler version of the preprocessing diagram."""
    print("Starting diagram creation...")
    
    # Create figure with simpler layout
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[1, 1.2])
    
    print("Creating data tables...")
    # Top left: Raw data tables
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.text(0.5, 0.5, 
             "NWM Data:\ninit_time, valid_time, flow, lead_time\n\n" + 
             "USGS Data:\ndatetime, flow",
             ha='center', va='center')
    ax1.set_title('1. Raw Input Data')
    
    print("Creating error calculation...")
    # Top right: Merged & error
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.text(0.5, 0.5, 
             "Merged Data:\ndatetime, nwm_flow, usgs_flow, lead_time\n\n" + 
             "Error = usgs_flow - nwm_flow",
             ha='center', va='center')
    ax2.set_title('2. Data Merge & Error Calculation')
    
    print("Creating sequence visualization...")
    # Bottom: Simple sequence diagram
    ax3 = fig.add_subplot(gs[1, :])
    
    # Simplified time series
    x = np.arange(30)
    y = np.sin(x/5) + 0.1*x
    
    # Basic plot
    ax3.plot(x, y, 'b-')
    
    # Simple windows
    ax3.axvspan(5, 5+10, alpha=0.2, color='blue', label='Input (24h)')
    ax3.axvspan(5+10, 5+10+8, alpha=0.2, color='green', label='Target (18h)')
    
    ax3.legend()
    ax3.set_title('3. Sequence Creation with Sliding Windows')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Value')
    
    # Add main title
    fig.suptitle('Data Preprocessing Pipeline', fontsize=16)
    
    print("Saving diagram...")
    # Save with lower DPI for speed
    output_path = 'preprocessing_pipeline_light.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Diagram saved as '{output_path}'")
    
    plt.close()  # Close the figure to save memory

if __name__ == "__main__":
    create_preprocessing_diagram_light()
    print("Done!")