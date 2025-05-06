"""
Utilities for GPU acceleration using TensorFlow Metal on Mac.
"""
import os
import tensorflow as tf

def configure_metal_gpu():
    """
    Configure TensorFlow to use Metal GPU on Mac.
    Returns True if Metal GPU is available and configured.
    """
    # Check if we're on macOS
    if os.path.exists('/System/Library/Frameworks/Metal.framework'):
        # List available devices
        physical_devices = tf.config.list_physical_devices()
        print("Available devices:")
        for device in physical_devices:
            print(f"  {device.name} - {device.device_type}")
        
        # Configure memory growth to prevent TF from allocating all GPU memory at once
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Metal GPU acceleration enabled with {len(gpus)} device(s)")
                
                # Print TensorFlow build info
                print(f"TensorFlow version: {tf.__version__}")
                print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
                print(f"TensorFlow GPU available: {tf.test.is_gpu_available()}")
                return True
            except RuntimeError as e:
                print(f"⚠️ Error configuring GPU: {e}")
    
    print("⚠️ Metal GPU not available. Using CPU only.")
    return False