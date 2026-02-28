import sys
import os

print("Python version:", sys.version)
print("CWD:", os.getcwd())

try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    import torch.nn as nn
    print("torch.nn imported successfully")
    
    # Try small operation
    x = torch.randn(1, 1, 128, 128)
    print("Tensor creation successful, shape:", x.shape)
    
except Exception as e:
    print("\n!!! TORCH IMPORT FAILED !!!")
    print("Error type:", type(e).__name__)
    print("Error message:", str(e))
    import traceback
    traceback.print_exc()

print("\nDiagnostics completed.")
