# mytorch

A from-scratch deep learning library implemented in NumPy. Every layer includes both a `forward()` and `backward()` pass, demonstrating how automatic differentiation and backpropagation work under the hood.

## What's here

```
mytorch/
├── nn/
│   ├── linear.py         # Fully-connected layer
│   ├── activation.py     # Identity, Sigmoid, Tanh, ReLU, GELU, Swish, Softmax
│   ├── batchnorm.py      # Batch normalization (1D)
│   ├── loss.py           # MSE and cross-entropy losses
│   ├── conv.py           # Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d
│   ├── pool.py           # MaxPool2d, MeanPool2d
│   ├── resampling.py     # Upsample / Downsample (1D and 2D)
│   ├── flatten.py        # Flatten spatial dims to vector
│   ├── rnn_cell.py       # Vanilla RNN cell with BPTT
│   ├── gru_cell.py       # GRU cell with full gate derivatives
│   └── attention.py      # Scaled dot-product & multi-head attention
├── optim/
│   └── sgd.py            # SGD with momentum
└── ctc/
    ├── loss.py           # CTC forward-backward algorithm & loss
    └── decoding.py       # Greedy CTC decoder

examples/
├── mlp.py                # MLP models using mytorch layers
└── rnn_classifier.py     # RNN phoneme classifier & GRU character predictor
```

## Requirements

- Python 3.8+
- NumPy
- SciPy (for `GELU` only)

## Usage

```python
import numpy as np
from mytorch.nn import Linear, ReLU, CrossEntropyLoss
