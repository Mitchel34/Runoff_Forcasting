import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Arrow
import matplotlib.gridspec as gridspec

def create_model_diagrams():
    """Create architecture diagrams for LSTM and Transformer models."""
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    
    # Create subplots for each model
    ax_lstm = fig.add_subplot(gs[0])
    ax_transformer = fig.add_subplot(gs[1])
    
    # Draw the model architectures
    draw_lstm_architecture(ax_lstm)
    draw_transformer_architecture(ax_transformer)
    
    # Add title
    fig.suptitle('Model Architectures for Runoff Error Correction', fontsize=20)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = 'model_architectures.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagrams saved as '{output_path}'")
    
    plt.show()

def draw_lstm_architecture(ax):
    """Draw LSTM model architecture."""
    # Turn off axis
    ax.axis('off')
    
    # Define box positions (x, y, width, height)
    input_pos = (0.5, 0.85, 0.8, 0.1)
    lstm_pos = (0.5, 0.5, 0.8, 0.2) 
    output_pos = (0.5, 0.15, 0.8, 0.1)
    
    # Colors
    input_color = '#D6EAF8'
    lstm_color = '#AED6F1'
    output_color = '#D6EAF8'
    
    # Draw boxes
    draw_box(ax, input_pos, "Input\n(24×36)", input_color)
    draw_box(ax, lstm_pos, "LSTM Layer\n\nStation 21609641: 128 units, 0.2 dropout\nStation 20380357: 192 units, 0.3 dropout", lstm_color)
    draw_box(ax, output_pos, "Dense Output\n(18)", output_color)
    
    # Draw arrows
    draw_arrow(ax, (input_pos[0], input_pos[1]-input_pos[3]/2), (lstm_pos[0], lstm_pos[1]+lstm_pos[3]/2))
    draw_arrow(ax, (lstm_pos[0], lstm_pos[1]-lstm_pos[3]/2), (output_pos[0], output_pos[1]+output_pos[3]/2))
    
    # Add LSTM cell details
    ax.text(0.2, 0.5, "Cell\nState", fontsize=8, ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    ax.text(0.2, 0.45, "Hidden\nState", fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    ax.text(0.8, 0.5, "Gates\n- Input\n- Forget\n- Output", fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    ax.set_title("LSTM Model Architecture", fontsize=16)

def draw_transformer_architecture(ax):
    """Draw Transformer model architecture."""
    # Turn off axis
    ax.axis('off')
    
    # Define box positions (x, y, width, height)
    input_pos = (0.5, 0.9, 0.8, 0.08)
    encoder_pos = (0.5, 0.62, 0.8, 0.2)
    pooling_pos = (0.5, 0.45, 0.8, 0.08)
    mlp_pos = (0.5, 0.3, 0.8, 0.08)
    output_pos = (0.5, 0.15, 0.8, 0.08)
    
    # Colors
    input_color = '#D6EAF8'
    encoder_color = '#AED6F1'
    pooling_color = '#D6EAF8'
    mlp_color = '#D6EAF8'
    output_color = '#D6EAF8'
    
    # Draw boxes
    draw_box(ax, input_pos, "Input\n(24×36)", input_color)
    draw_box(ax, encoder_pos, "Transformer Encoder Blocks\n\nStation 21609641: 6 blocks, 4 heads, FF dim 64\nStation 20380357: 4 blocks, 8 heads, FF dim 192", encoder_color)
    draw_box(ax, pooling_pos, "Global Average Pooling", pooling_color)
    draw_box(ax, mlp_pos, "MLP Head", mlp_color)
    draw_box(ax, output_pos, "Dense Output\n(18)", output_color)
    
    # Draw arrows
    draw_arrow(ax, (input_pos[0], input_pos[1]-input_pos[3]/2), (encoder_pos[0], encoder_pos[1]+encoder_pos[3]/2))
    draw_arrow(ax, (encoder_pos[0], encoder_pos[1]-encoder_pos[3]/2), (pooling_pos[0], pooling_pos[1]+pooling_pos[3]/2))
    draw_arrow(ax, (pooling_pos[0], pooling_pos[1]-pooling_pos[3]/2), (mlp_pos[0], mlp_pos[1]+mlp_pos[3]/2))
    draw_arrow(ax, (mlp_pos[0], mlp_pos[1]-mlp_pos[3]/2), (output_pos[0], output_pos[1]+output_pos[3]/2))
    
    # Add transformer details
    ax.text(0.2, 0.62, "Multi-Head\nAttention", fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    ax.text(0.8, 0.62, "Feed\nForward", fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    ax.set_title("Transformer Model Architecture", fontsize=16)

def draw_box(ax, position, text, facecolor):
    """Draw a box with text."""
    x, y, width, height = position
    x_min, x_max = x - width/2, x + width/2
    y_min, y_max = y - height/2, y + height/2
    
    # Create rectangle
    rect = Rectangle((x_min, y_min), width, height,
                    facecolor=facecolor, edgecolor='black', alpha=0.8)
    ax.add_patch(rect)
    
    # Add text
    ax.text(x, y, text, ha='center', va='center', fontsize=10)

def draw_arrow(ax, start, end):
    """Draw an arrow between two points."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", linewidth=1.5))

# Add explanation text below the diagrams
def add_explanation(fig):
    """Add explanation text at the bottom of the figure."""
    text = ("LSTM processes sequences step-by-step through its gates, while Transformer analyzes entire sequences at once via self-attention.\n"
            "Note how the challenging station (20380357) required larger models with different configurations.")
    fig.text(0.5, 0.01, text, ha='center', fontsize=11, wrap=True)

if __name__ == "__main__":
    create_model_diagrams()