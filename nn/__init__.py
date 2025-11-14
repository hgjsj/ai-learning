"""Neural network demo package.

The actual PyTorch example implementation lives in `nn.demo`.
This file simply re-exports the public API for convenience so you can write:
    from nn import run_demo, DemoConfig
"""
from .neural_network import *  # noqa: F401,F403
