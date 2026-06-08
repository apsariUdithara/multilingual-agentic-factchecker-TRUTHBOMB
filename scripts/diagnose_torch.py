import sys
try:
    import torch
    print(f"Torch imported from: {torch.__file__ if hasattr(torch, '__file__') else 'Unknown'}")
    print(f"Torch version: {getattr(torch, '__version__', 'No __version__ attribute')}")
    print(f"Torch path: {getattr(torch, '__path__', 'No __path__ attribute')}")
    print(f"Torch dir: {dir(torch)[:10]}...")
except Exception as e:
    print(f"Failed to import torch: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"Failed to import transformers: {e}")
