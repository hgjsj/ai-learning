# ai-learning

A minimal playground project that now includes a simple PyTorch neural network demo.

## Features
- Synthetic regression dataset generator
- Small MLP model (2 hidden layers) for regression
- Training loop with validation loss reporting
- Quick sample predictions after training

## Installation
Ensure you have Python 3.11+.

```bash
pip install -e .
```
(Or use your preferred environment / dependency manager.)

## Run the basic greeting
```bash
python main.py
```

## Run the neural network demo
```bash
python main.py --demo --epochs 3
```
If you omit `--epochs`, it defaults to 5.

## Run the demo module directly
```bash
python -m nn.demo --epochs 3
```

## Modifying the demo
Open `nn/demo.py` and tweak:
- `DemoConfig` for model sizes, learning rate, data sizes
- `build_model` to change architecture
- `make_synthetic_regression` to alter the underlying function

## API Usage
You can also import and run programmatically:
```python
from nn import DemoConfig, train_demo
results = train_demo(DemoConfig(epochs=2))
print(results['final_val_loss'])
```

Enjoy experimenting!

