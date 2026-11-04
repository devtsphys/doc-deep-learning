# Deep Learning Architectures: Reference Card & Cheat Sheet (PyTorch)

> Covers: MLP · CNN · RNN · LSTM · Transformer · GAN · Autoencoder · GNN
> Each chapter: **Basics → Concepts → Architecture diagram → Process → Use cases → Pros/Cons → PyTorch code**

---

## Table of Contents
0. [Fundamentals & Training Loop](#0-fundamentals--training-loop)
1. [Feedforward NN (MLP)](#1-feedforward-neural-networks-mlp)
2. [Convolutional NN (CNN)](#2-convolutional-neural-networks-cnn)
3. [Recurrent NN (RNN)](#3-recurrent-neural-networks-rnn)
4. [LSTM](#4-long-short-term-memory-lstm)
5. [Transformer](#5-transformer-networks)
6. [GAN](#6-generative-adversarial-networks-gan)
7. [Autoencoders](#7-autoencoders)
8. [Graph Neural Networks](#8-graph-neural-networks-gnn)
9. [Comparison Matrix & Chooser](#9-comparison-matrix--architecture-chooser)
10. [Practical Cheat Sheet](#10-practical-cheat-sheet)

---

## 0. Fundamentals & Training Loop

### Core vocabulary
| Term | Meaning |
|---|---|
| **Tensor** | n-dimensional array; the basic data structure (`torch.Tensor`) |
| **Parameter** | Learnable weight/bias, updated by the optimizer |
| **Hyperparameter** | Set by you: learning rate, depth, batch size, dropout, ... |
| **Forward pass** | Compute predictions from inputs |
| **Loss** | Scalar measuring prediction error |
| **Backpropagation** | Chain rule applied to get dLoss/dParam for every parameter |
| **Gradient descent** | `θ ← θ − η · ∇θ L` |
| **Epoch / Batch** | One full pass over data / a subset processed at once |
| **Overfitting** | Great train score, poor validation score |

### Activation functions
| Name | Formula | Notes |
|---|---|---|
| ReLU | `max(0,x)` | Default for hidden layers; "dying ReLU" risk |
| LeakyReLU | `max(αx,x)` | Fixes dead neurons |
| GELU | `x·Φ(x)` | Standard in Transformers |
| SiLU/Swish | `x·σ(x)` | Smooth, popular in modern CNNs/LLMs |
| Sigmoid | `1/(1+e^-x)` | Binary output; vanishing gradients |
| Tanh | `(e^x−e^-x)/(e^x+e^-x)` | Zero-centered; used in RNN/LSTM |
| Softmax | `e^xi / Σe^xj` | Multi-class probabilities |

### Loss functions
| Task | Loss | PyTorch |
|---|---|---|
| Regression | MSE / MAE / Huber | `nn.MSELoss`, `nn.L1Loss`, `nn.HuberLoss` |
| Binary classification | BCE | `nn.BCEWithLogitsLoss` (takes raw logits) |
| Multi-class | Cross-entropy | `nn.CrossEntropyLoss` (takes raw logits, int labels) |
| Reconstruction | MSE / BCE | see Autoencoders |

### Optimizers
`SGD(+momentum)` · `Adam` (default choice, lr≈1e-3) · `AdamW` (Adam + decoupled weight decay; Transformers) · `RMSprop` (RNNs/GANs)

### Universal training loop template
```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()                                  # enables dropout, batchnorm training mode
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)      # 1. reset gradients
        pred = model(x)                            # 2. forward
        loss = loss_fn(pred, y)                    # 3. loss
        loss.backward()                            # 4. backprop
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # (optional) stabilise
        optimizer.step()                           # 5. update weights
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()                                   # no gradient tracking for evaluation
def evaluate(model, loader, loss_fn):
    model.eval()                                   # disables dropout, uses BN running stats
    total, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total += loss_fn(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total / n, correct / n

# Usage
# model = MyModel().to(device)
# opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
# sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
# for epoch in range(epochs):
#     tr = train_one_epoch(model, train_loader, nn.CrossEntropyLoss(), opt)
#     va_loss, va_acc = evaluate(model, val_loader, nn.CrossEntropyLoss())
#     sched.step()
```

### Regularization toolbox
`Dropout` · `Weight decay` · `BatchNorm / LayerNorm` · `Data augmentation` · `Early stopping` · `Label smoothing` · `Gradient clipping`

---

## 1. Feedforward Neural Networks (MLP)

### Basics
The simplest deep network: layers of fully connected ("dense") neurons. Information flows **one way** (input → hidden → output), with no loops. Each neuron computes `a = σ(w·x + b)`. By the **Universal Approximation Theorem**, an MLP with one sufficiently wide hidden layer can approximate any continuous function; depth makes this more efficient.

### Key concepts
- **Dense / Linear layer**: `y = xWᵀ + b`
- **Non-linearity**: without activations, stacked layers collapse into one linear map
- **Width vs depth**: wider = more capacity per layer; deeper = more abstract features
- **Weight init**: Xavier/Glorot (tanh/sigmoid), Kaiming/He (ReLU)
- **Input must be flat**: images/sequences are flattened (loses spatial/temporal structure)

### Architecture
```
 Input layer      Hidden layer 1    Hidden layer 2     Output layer
   (x1) ○──┐        ┌──○ (h1)         ┌──○ (h1')         ┌──○ (y1)
   (x2) ○──┼───────►├──○ (h2) ───────►├──○ (h2') ───────►├──○ (y2)
   (x3) ○──┼  W1,b1 ├──○ (h3)  W2,b2  ├──○ (h3')  W3,b3  └──○ (y3)
   (x4) ○──┘        └──○ (h4)         └──○ (h4')
        every neuron connects to every neuron of the next layer
        each layer:  h = activation(W·x + b)
```

### Process / functionality
1. **Forward**: `h1 = ReLU(W1x+b1)`, `h2 = ReLU(W2h1+b2)`, `ŷ = W3h2+b3`
2. **Loss**: compare `ŷ` with `y`
3. **Backward**: gradients flow output → input via chain rule
4. **Update**: optimizer adjusts all `W`, `b`

### Use cases
Tabular/structured data (credit scoring, churn) · simple regression/classification · embedding heads and classifier heads in bigger networks · feed-forward blocks inside Transformers · function approximation (e.g. physics/RL value networks)

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---|---|
| Simple, fast, easy to debug | Ignores spatial/temporal structure |
| Works on any fixed-size vector | Parameter count explodes with input size |
| Universal approximator | Fixed input size |
| Strong baseline for tabular data | Often beaten by gradient boosting on small tabular data |

### PyTorch code
```python
import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(256, 128), out_dim=10, p_drop=0.2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))      # raw logits, no softmax
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x.flatten(1))                # flatten e.g. 28x28 image -> 784

# --- Usage: MNIST-like classification ---
model = MLP(in_dim=784, hidden_dims=(512, 256), out_dim=10)
x = torch.randn(32, 1, 28, 28)
logits = model(x)                                    # (32, 10)
probs = logits.softmax(dim=1)
pred = probs.argmax(dim=1)
print(logits.shape, sum(p.numel() for p in model.parameters()), "params")

# --- Regression variant: set out_dim=1 and use nn.MSELoss ---
```

---

## 2. Convolutional Neural Networks (CNN)

### Basics
CNNs exploit **grid-like structure** (images, audio spectrograms, 1D signals). Instead of connecting everything to everything, small **learnable filters (kernels)** slide across the input, detecting local patterns (edges → textures → parts → objects).

### Key concepts
- **Convolution**: dot product of kernel with local patch → **feature map**
- **Local connectivity + weight sharing**: same kernel reused across positions → far fewer parameters, **translation equivariance**
- **Kernel size, stride, padding, dilation, channels**
- **Output size**: `O = ⌊(W − K + 2P)/S⌋ + 1`
- **Pooling** (max/avg): downsample, add small translation invariance
- **Receptive field**: input region influencing one output unit; grows with depth
- **BatchNorm**, **Residual connections** (ResNet: `y = F(x) + x`), **1×1 conv** (channel mixing), **depthwise-separable conv** (MobileNet efficiency)
- **Global average pooling** replaces big flatten+FC layers

### Architecture
```
Input image        Conv+ReLU        Pool        Conv+ReLU       Pool      Flatten   FC      Output
┌─────────┐      ┌──────────┐    ┌───────┐    ┌─────────┐    ┌──────┐
│ 3×32×32 │ ───► │ 32×32×32 │ ─► │32×16×16│ ─►│64×16×16 │ ─► │64×8×8│ ─► [4096] ─► [128] ─► [10]
└─────────┘      └──────────┘    └───────┘    └─────────┘    └──────┘
   RGB           3×3 kernels     maxpool 2×2   3×3 kernels    maxpool        dense   dense   classes
  
  Feature hierarchy:  edges/colors  →  textures/shapes  →  object parts  →  objects

 Kernel sliding (3×3, stride 1):
   ┌─┬─┬─┐
   │ │ │ │──► Σ(patch ⊙ kernel) + b ──► one output pixel
   └─┴─┴─┘   moves right/down over the entire input
```

### Process / functionality
1. Conv layers extract local features (many filters → many channels)
2. Activation + (BatchNorm) adds non-linearity and stabilises training
3. Pooling/stride reduces spatial size, increases channel depth
4. Classifier head (GAP + Linear) maps features to outputs
5. Backprop learns the kernel values

### Use cases
Image classification · object detection (YOLO, Faster R-CNN) · semantic segmentation (U-Net) · medical imaging · OCR · video (3D conv) · audio/spectrograms · 1D CNN for time series & text · feature extractor for transfer learning

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---|---|
| Few parameters via weight sharing | Limited global context (needs depth) |
| Excellent for spatial data | Not rotation/scale invariant by default |
| Translation equivariance | Needs lots of data/augmentation |
| Mature ecosystem, pretrained models | Pooling can lose fine detail |
| Efficient on GPU/edge | Vision Transformers now often surpass on large data |

### PyTorch code
```python
import torch, torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, 32), ConvBlock(32, 32), nn.MaxPool2d(2),   # 32 -> 16
            ConvBlock(32, 64),    ConvBlock(64, 64),  nn.MaxPool2d(2),  # 16 -> 8
            ConvBlock(64, 128),   nn.AdaptiveAvgPool2d(1),              # -> 128x1x1
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN()
print(model(torch.randn(8, 3, 32, 32)).shape)        # (8, 10)

# --- Residual block (ResNet style) ---
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
    def forward(self, x): return torch.relu(x + self.f(x))     # skip connection

# --- Transfer learning with a pretrained model ---
from torchvision import models
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in resnet.parameters(): p.requires_grad = False          # freeze backbone
resnet.fc = nn.Linear(resnet.fc.in_features, 5)                # new trainable head (5 classes)

# --- Typical augmentation ---
from torchvision import transforms as T
train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
```

---

## 3. Recurrent Neural Networks (RNN)

### Basics
RNNs process **sequences** (text, time series, audio) one step at a time while carrying a **hidden state** — a memory summarising what was seen so far. The *same weights* are reused at every time step.

### Key concepts
- **Hidden state** `h_t = tanh(W_xh·x_t + W_hh·h_{t−1} + b)`
- **Parameter sharing across time** → handles variable-length input
- **Unrolling**: the loop viewed as a deep network over time
- **BPTT** (Backprop Through Time): gradients flow back through all steps
- **Vanishing/exploding gradients**: repeated multiplication by `W_hh` → long-range dependencies are hard to learn. Mitigation: gradient clipping, LSTM/GRU
- **Modes**: many-to-one (sentiment), one-to-many (captioning), many-to-many (translation, tagging)
- **Bidirectional RNN**: reads sequence forward and backward
- **Stacked RNN**: multiple layers; **GRU**: simplified gated variant of LSTM

### Architecture
```
 Folded:                     Unrolled over time:

      ┌───┐                 x₁         x₂         x₃         x₄
 x_t─►│ h │─► y_t            │          │          │          │
      └─▲─┘                  ▼          ▼          ▼          ▼
        └─ loop           ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
        (h_{t-1})   h₀──► │ RNN │─h₁►│ RNN │─h₂►│ RNN │─h₃►│ RNN │─► h₄
                          └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘
                             ▼          ▼          ▼          ▼
                             y₁         y₂         y₃         y₄
        (same weights W_xh, W_hh, W_hy in every cell)

 h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)      y_t = W_hy h_t + b_y
```

### Process / functionality
1. Initialise `h₀ = 0`
2. For each step: combine current input with previous state → new state
3. Output per step (`y_t`) or only at the end (`h_T`)
4. Loss over outputs; **BPTT** computes gradients; clip gradient norm
5. Update shared weights

### Use cases
Simple time-series forecasting · character-level language models · sensor data · basic sequence labelling · small-scale speech tasks. *(For anything with long dependencies prefer LSTM/GRU/Transformer.)*

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---|---|
| Natural fit for sequences, variable length | Vanishing/exploding gradients |
| Compact: weights shared over time | Short memory in practice |
| Constant model size regardless of length | **Sequential**, so no parallelism over time → slow training |
| Streaming/online inference possible | Largely superseded by LSTM/Transformers |

### PyTorch code
```python
import torch, torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, bidirectional=False):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, nonlinearity="tanh",
                          dropout=0.2 if num_layers > 1 else 0.0,
                          bidirectional=bidirectional)
        d = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * d, num_classes)

    def forward(self, x):                       # x: (batch, seq_len, input_size)
        out, h_n = self.rnn(x)                  # out: (B, T, H*d)   h_n: (layers*d, B, H)
        return self.fc(out[:, -1])              # many-to-one: last time step

model = RNNClassifier(input_size=8)
print(model(torch.randn(16, 50, 8)).shape)      # (16, 3)

# --- Manual RNN cell (to understand the math) ---
class ManualRNN(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.Wx = nn.Linear(in_dim, hid)
        self.Wh = nn.Linear(hid, hid, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.Wh.in_features, device=x.device)
        outs = []
        for t in range(T):
            h = torch.tanh(self.Wx(x[:, t]) + self.Wh(h))
            outs.append(h)
        return torch.stack(outs, dim=1), h

# --- Variable-length sequences: pack to ignore padding ---
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
seqs = [torch.randn(5, 8), torch.randn(3, 8)]
lengths = torch.tensor([5, 3])
padded = pad_sequence(seqs, batch_first=True)                   # (2, 5, 8)
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
out, h = nn.RNN(8, 16, batch_first=True)(packed)
out, _ = pad_packed_sequence(out, batch_first=True)

# --- Always clip gradients when training RNNs ---
# loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
```

---

## 4. Long Short-Term Memory (LSTM)

### Basics
LSTM (Hochreiter & Schmidhuber, 1997) is an RNN variant built to learn **long-range dependencies**. It adds a separate **cell state** `c_t` (a "conveyor belt" of memory) regulated by **gates** that decide what to forget, write and expose.

### Key concepts
- **Cell state `c_t`**: long-term memory, updated additively → gradients flow much better
- **Hidden state `h_t`**: short-term/output memory
- **Forget gate `f_t`**: what to erase from `c`
- **Input gate `i_t`** + **candidate `g_t`**: what new info to write
- **Output gate `o_t`**: what part of `c` to expose as `h`
- **GRU**: 2 gates (reset, update), no separate cell; fewer params, similar performance
- Tricks: forget-gate bias init to 1, bidirectional, stacking, attention on top, gradient clipping

### Equations
```
f_t = σ(W_f·[h_{t-1}, x_t] + b_f)        forget gate
i_t = σ(W_i·[h_{t-1}, x_t] + b_i)        input gate
g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)     candidate memory
o_t = σ(W_o·[h_{t-1}, x_t] + b_o)        output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          new cell state
h_t = o_t ⊙ tanh(c_t)                    new hidden state
```

### Architecture
```
                    ┌──────────────── LSTM cell ────────────────┐
  c_{t-1} ─────────►│──────(×)──────────────(+)─────────────────│─────────► c_t
   (long-term)      │       ▲ f_t            ▲                  │       │
                    │       │                │ (×)              │     tanh
                    │       │            i_t ─┴─ g_t            │       │
                    │       │             ▲      ▲              │       ▼
                    │      [σ]           [σ]   [tanh]    [σ]──►(×)◄────┘
                    │       ▲             ▲      ▲        o_t     │
  h_{t-1} ─────────►│───────┴─────────────┴──────┴────────┘       │
   (short-term)     │                  concat[h_{t-1}, x_t]       ├────────► h_t
  x_t ─────────────►│                                             │
                    └─────────────────────────────────────────────┘

  forget: "what do I drop?"  input: "what do I store?"  output: "what do I reveal?"
```

### Process / functionality
1. Concatenate `h_{t−1}` and `x_t`
2. Compute the four gates
3. Update memory: forget old (`f⊙c`) + add new (`i⊙g`)
4. Output filtered memory as `h_t`
5. Repeat per step; BPTT trains all gate weights (use clipping)

### Use cases
Language modelling & NER (pre-Transformer) · speech recognition · time-series forecasting & anomaly detection · machine translation (seq2seq) · handwriting recognition · music generation · predictive maintenance · sensor/IoT streams

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---|---|
| Captures long dependencies (hundreds of steps) | Still sequential, so slow to train |
| Mitigates vanishing gradients | ~4× parameters of a plain RNN |
| Works with variable-length input | Struggles at very long ranges (1000s of tokens) |
| Good with small/medium sequence data | Outperformed by Transformers on large NLP |
| Cheap streaming inference (constant memory) | Many hyperparameters |

### PyTorch code
```python
import torch, torch.nn as nn

class LSTMForecaster(nn.Module):
    """Many-to-one: predict next value from a window of past values."""
    def __init__(self, n_features=1, hidden=64, layers=2, out_dim=1, bidir=False):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=0.2 if layers > 1 else 0, bidirectional=bidir)
        self.head = nn.Sequential(nn.Linear(hidden * (2 if bidir else 1), 32), nn.ReLU(),
                                  nn.Linear(32, out_dim))
    def forward(self, x):                          # (B, T, F)
        out, (h_n, c_n) = self.lstm(x)             # out: (B, T, H)
        return self.head(out[:, -1])

model = LSTMForecaster()
print(model(torch.randn(16, 30, 1)).shape)         # (16, 1)

# --- Sliding-window dataset for time series ---
def make_windows(series, window=30, horizon=1):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i+window]); y.append(series[i+window+horizon-1])
    return torch.tensor(X).unsqueeze(-1).float(), torch.tensor(y).unsqueeze(-1).float()

# --- Text classifier with embeddings ---
class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=128, classes=2, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, classes)
    def forward(self, tokens):                     # tokens: (B, T) int64
        _, (h_n, _) = self.lstm(self.emb(tokens))  # h_n: (2, B, H)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)   # concat last fwd + bwd states
        return self.fc(h)

# --- Seq2seq decoder step with explicit LSTMCell (autoregressive generation) ---
cell = nn.LSTMCell(input_size=16, hidden_size=32)
h = c = torch.zeros(4, 32)
x = torch.randn(4, 16)
for _ in range(5):
    h, c = cell(x, (h, c))

# --- GRU drop-in replacement ---
gru = nn.GRU(8, 32, batch_first=True)             # returns (out, h_n) only
```

---

## 5. Transformer Networks

### Basics
Introduced in *"Attention Is All You Need"* (2017). Transformers drop recurrence and process the **whole sequence in parallel** using **self-attention**, where each token directly looks at every other token. They underpin LLMs (GPT, Llama), BERT, ViT, Whisper, and more.

### Key concepts
- **Token embedding + positional encoding** (sinusoidal, learned, RoPE, ALiBi): attention is order-blind, so position must be injected
- **Self-attention**: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`
  - Q = what I look for, K = what I offer, V = the content
- **Multi-head attention**: several attention "views" in parallel, then concatenated
- **Masking**: *padding mask* (ignore pads), *causal mask* (no peeking at future tokens)
- **Feed-forward block** (position-wise MLP, usually 4× width, GELU)
- **Residual connections + LayerNorm** (Pre-LN is more stable)
- **Variants**: Encoder-only (BERT; understanding) · Decoder-only (GPT; generation) · Encoder–Decoder (T5, original; translation) · ViT (image patches as tokens)
- **Complexity**: attention is O(n²) in sequence length → FlashAttention, sparse/linear attention, KV-cache
- **Training recipe**: AdamW, warmup + cosine decay, dropout, label smoothing, mixed precision

### Architecture
```
            ENCODER (×N)                                DECODER (×N)
   Input tokens                                   Output tokens (shifted right)
        │                                                  │
   Embedding + Positional Enc.                    Embedding + Positional Enc.
        │                                                  │
 ┌──────▼───────────────────┐                ┌──────────────▼───────────────┐
 │ Multi-Head Self-Attention│                │ MASKED Multi-Head Self-Attn  │
 │   + Add & LayerNorm      │                │   + Add & LayerNorm          │
 ├──────────────────────────┤                ├──────────────────────────────┤
 │ Feed-Forward (MLP)       │   K, V         │ Multi-Head CROSS-Attention   │
 │   + Add & LayerNorm      ├───────────────►│  (Q from decoder, K/V enc.)  │
 └──────┬───────────────────┘                │   + Add & LayerNorm          │
        │ (×N)                               ├──────────────────────────────┤
        ▼                                    │ Feed-Forward + Add & Norm    │
   encoder output                            └──────────────┬───────────────┘
                                                            │ (×N)
                                                   Linear → Softmax
                                                            │
                                                  next-token probabilities

 Scaled dot-product attention:
   X ─► Wq ─► Q ┐
   X ─► Wk ─► K ┼─► scores = QKᵀ/√d ─► (mask) ─► softmax ─► weights ─► ×V ─► output
   X ─► Wv ─► V ┘
```

### Process / functionality
1. Tokenize text → IDs → embeddings; add positions
2. **Self-attention**: every token computes weights over all tokens, builds a context-aware vector
3. **FFN** transforms each position independently
4. Stack N layers (residual + norm around each sublayer)
5. **Training**: next-token prediction (GPT), masked-token (BERT), seq2seq (T5) via cross-entropy
6. **Inference**: autoregressive decoding (greedy, beam, top-k/top-p, temperature) with KV-cache

### Use cases
LLMs/chatbots · translation · summarisation · code generation · question answering/RAG · embeddings & search · vision (ViT, DETR) · speech (Whisper) · protein folding (AlphaFold) · multimodal models (CLIP, VLMs) · time series (Informer, PatchTST)

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---|---|
| Fully parallel training; scales superbly | O(n²) memory/compute in sequence length |
| Captures global/long-range context directly | Needs large data & compute |
| Transfer learning (pretrain → fine-tune) | No built-in notion of order (needs pos. enc.) |
| Versatile across modalities | Hallucinations, hard to interpret |
| Strong state of the art everywhere | Expensive inference for large models |

### PyTorch code: from-scratch building blocks
```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk, self.causal = n_heads, d_model // n_heads, causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):                 # x: (B, T, D); pad_mask: (B, T) True=pad
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v)]  # (B,h,T,dk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)                        # (B,h,T,T)
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float("-inf"))
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(y)
        # Fast path in PyTorch >= 2: F.scaled_dot_product_attention(q, k, v, is_causal=True)

class TransformerBlock(nn.Module):                        # Pre-LN
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, causal=False):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, causal)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))
    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        return x + self.ffn(self.ln2(x))

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[: x.size(1)]

class MiniGPT(nn.Module):
    """Decoder-only language model."""
    def __init__(self, vocab, d_model=256, n_heads=4, n_layers=4, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = SinusoidalPE(d_model, max_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, causal=True) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight                # weight tying
    def forward(self, idx):
        x = self.pos(self.tok(idx))
        for b in self.blocks: x = b(x)
        return self.head(self.ln(x))                      # (B, T, vocab)

    @torch.no_grad()
    def generate(self, idx, max_new=50, temperature=1.0, top_k=40):
        for _ in range(max_new):
            logits = self(idx[:, -512:])[:, -1] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

model = MiniGPT(vocab=1000)
x = torch.randint(0, 1000, (2, 16))
logits = model(x)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, 1000), x[:, 1:].reshape(-1))  # next-token loss
print(logits.shape, loss.item())
```

### PyTorch code: built-in modules & Hugging Face
```python
# --- Built-in encoder (e.g. classification / time series) ---
layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512,
                                   dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=4)
x = torch.randn(8, 20, 128)
out = encoder(x, src_key_padding_mask=None)           # (8, 20, 128)
cls_logits = nn.Linear(128, 3)(out.mean(dim=1))       # mean pooling -> classes

# --- Built-in full encoder-decoder ---
tr = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
tgt_mask = nn.Transformer.generate_square_subsequent_mask(10)
y = tr(torch.randn(4, 20, 128), torch.randn(4, 10, 128), tgt_mask=tgt_mask)

# --- Pretrained models with Hugging Face (pip install transformers) ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
clf = pipeline("sentiment-analysis")                                  # 1-liner inference
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
mdl = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
batch = tok(["great movie", "awful"], padding=True, truncation=True, return_tensors="pt")
logits = mdl(**batch).logits

# --- Vision Transformer patch embedding (core idea of ViT) ---
class PatchEmbed(nn.Module):
    def __init__(self, img=224, patch=16, ch=3, d=768):
        super().__init__()
        self.proj = nn.Conv2d(ch, d, kernel_size=patch, stride=patch)   # patchify + linear
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, d)
```

---

## 6. Generative Adversarial Networks (GAN)

### Basics
Introduced by Goodfellow et al. (2014). Two networks play a **minimax game**: a **Generator (G)** creates fake data from random noise, a **Discriminator (D)** tells real from fake. G learns to fool D; D learns not to be fooled. At equilibrium G's output distribution matches the real data distribution.

### Key concepts
- **Latent vector `z`** ~ N(0, I): random seed for generation
- **Minimax objective**: `min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]`
- **Non-saturating G loss**: G maximises `log D(G(z))` (better gradients early)
- **Mode collapse**: G produces only few varieties
- **Training instability**: oscillation, vanishing gradients if D too strong
- **Stabilisers**: DCGAN guidelines (strided convs, BatchNorm, LeakyReLU/ReLU, tanh output), **WGAN / WGAN-GP** (Wasserstein distance + gradient penalty), spectral norm, label smoothing, two-timescale updates (TTUR), R1 penalty
- **Variants**: DCGAN · cGAN (conditional on label) · Pix2Pix/CycleGAN (image-to-image) · StyleGAN (high-res faces) · BigGAN · SRGAN (super-resolution)
- **Metrics**: FID, Inception Score, precision/recall

### Architecture
```
                    ┌──────────────┐
  noise z ~ N(0,I)─►│  GENERATOR   │──► fake image G(z) ──┐
  (e.g. 100-dim)    │ (upsampling: │                      │      ┌───────────────┐
                    │  ConvTrans.) │                      ├────► │ DISCRIMINATOR │──► P(real)
                    └──────▲───────┘                      │      │ (downsampling │    in [0,1]
                           │                              │      │   CNN)        │
                           │                real image ───┘      └───────┬───────┘
                           │                  x ~ data                   │
                           │                                             │
                           └──── gradient of D's verdict (G loss) ◄──────┤
                                                                         │
                                       D loss: real→1, fake→0 ◄──────────┘

  Generator:  z[100] → FC → 256×4×4 → ConvT → 128×8×8 → ConvT → 64×16×16 → ConvT → 1×32×32 (tanh)
  Discr.:     1×32×32 → Conv → 64×16×16 → Conv → 128×8×8 → Conv → 256×4×4 → FC → 1 (logit)
```

### Process / functionality (alternating training)
1. **Train D**: batch of real (label 1) + batch of fakes `G(z).detach()` (label 0); minimise BCE; step D
2. **Train G**: generate fakes, ask D, loss = BCE(D(G(z)), **1**) (G wants "real"); step G
3. Repeat; monitor generated samples visually and with FID
4. **Inference**: sample `z`, run only G

### Use cases
Photorealistic image synthesis · data augmentation / synthetic data · super-resolution · image-to-image translation (style transfer, sketch→photo) · inpainting · deepfakes/face editing · anomaly detection (AnoGAN) · drug/molecule design · game asset generation. *(Diffusion models now lead in image generation quality/stability.)*

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---|---|
| Very sharp, realistic samples | Unstable, hard-to-tune training |
| Fast single-pass sampling | Mode collapse |
| No explicit likelihood needed | No likelihood/density → hard to evaluate |
| Flexible, many variants | Needs lots of data; ethical risk (deepfakes) |

### PyTorch code (DCGAN for 32×32 grayscale)
```python
import torch, torch.nn as nn

Z_DIM = 100

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, ch=64, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ch*4, 4, 1, 0, bias=False), nn.BatchNorm2d(ch*4), nn.ReLU(True),  # 4x4
            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1, bias=False),  nn.BatchNorm2d(ch*2), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(ch*2, ch, 4, 2, 1, bias=False),    nn.BatchNorm2d(ch),   nn.ReLU(True),  # 16x16
            nn.ConvTranspose2d(ch, out_ch, 4, 2, 1),              nn.Tanh())                            # 32x32
    def forward(self, z): return self.net(z.view(z.size(0), -1, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, in_ch=1, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1),          nn.LeakyReLU(0.2, True),                         # 16
            nn.Conv2d(ch, ch*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ch*2), nn.LeakyReLU(0.2, True),  # 8
            nn.Conv2d(ch*2, ch*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ch*4), nn.LeakyReLU(0.2, True),# 4
            nn.Conv2d(ch*4, 1, 4, 1, 0))                                                              # 1x1 logit
    def forward(self, x): return self.net(x).view(-1)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)) and hasattr(m, "weight"):
        nn.init.normal_(m.weight, 0.0, 0.02)

device = "cuda" if torch.cuda.is_available() else "cpu"
G, D = Generator().to(device), Discriminator().to(device)
G.apply(init_weights); D.apply(init_weights)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()

def train_step(real):                                  # real: (B,1,32,32) scaled to [-1, 1]
    B = real.size(0)
    ones, zeros = torch.ones(B, device=device), torch.zeros(B, device=device)
    # ---- 1) Discriminator ----
    z = torch.randn(B, Z_DIM, device=device)
    fake = G(z)
    loss_D = bce(D(real), ones * 0.9) + bce(D(fake.detach()), zeros)   # 0.9 = one-sided label smoothing
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()
    # ---- 2) Generator ----
    loss_G = bce(D(fake), ones)                         # fool D: fake should be classified as real
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    return loss_D.item(), loss_G.item()

# --- Sampling after training ---
G.eval()
with torch.no_grad():
    imgs = G(torch.randn(16, Z_DIM, device=device)).cpu()     # (16,1,32,32) in [-1,1]
    imgs = (imgs + 1) / 2                                      # -> [0,1]

# --- Conditional GAN trick: concatenate label embedding to z ---
# z_cond = torch.cat([z, nn.Embedding(10, 10)(labels)], dim=1)
# --- WGAN-GP gradient penalty ---
def gradient_penalty(D, real, fake):
    a = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    mix = (a * real + (1 - a) * fake).requires_grad_(True)
    g = torch.autograd.grad(D(mix).sum(), mix, create_graph=True)[0]
    return ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()
```

---

## 7. Autoencoders

### Basics
An autoencoder (AE) is trained to **reconstruct its own input** through a narrow **bottleneck**. This forces it to learn a compact, meaningful representation (the **latent code**) in an unsupervised/self-supervised way.

### Key concepts
- **Encoder** `z = f(x)` compresses; **Decoder** `x̂ = g(z)` reconstructs
- **Bottleneck / latent space**: dimension smaller than input (undercomplete)
- **Reconstruction loss**: MSE (continuous) or BCE (inputs in [0,1])
- **Variants**
  | Type | Idea |
  |---|---|
  | Vanilla / Undercomplete | Small bottleneck forces compression |
  | Sparse | L1/KL penalty on activations |
  | **Denoising (DAE)** | Input corrupted with noise, target = clean input |
  | Contractive | Penalise Jacobian → robust codes |
  | **Variational (VAE)** | Latent is a *distribution* `N(μ,σ²)`; loss = recon + KL; enables **generation** |
  | VQ-VAE | Discrete latent codebook (basis of many image/audio generators) |
  | Convolutional AE | Conv encoder/decoder for images |
- **VAE reparameterisation trick**: `z = μ + σ ⊙ ε`, `ε ~ N(0,I)` → differentiable sampling
- **VAE loss (ELBO)**: `L = ‖x − x̂‖² + β · KL(N(μ,σ²) ‖ N(0,I))`,  `KL = −½ Σ(1 + log σ² − μ² − σ²)`

### Architecture
```
 Plain Autoencoder
   Input x          ENCODER               BOTTLENECK            DECODER           Output x̂
 ┌─────────┐   ┌───────────────┐         ┌───────┐         ┌───────────────┐    ┌─────────┐
 │  784    │──►│ 512 → 128 → 32│────────►│ z (16)│────────►│ 32 → 128 → 512│───►│  784    │
 └─────────┘   └───────────────┘         └───────┘         └───────────────┘    └─────────┘
                    (funnel ▽)          latent code             (funnel △)        loss = ‖x − x̂‖²

 Variational Autoencoder (VAE)
                                 ┌──► μ ────┐
   x ──► Encoder ──► features ──►│          ├─► z = μ + σ·ε ──► Decoder ──► x̂
                                 └──► log σ²┘        ▲
                                                     ε ~ N(0, I)
   Loss = Reconstruction(x, x̂) + β·KL(q(z|x) ‖ N(0,I))

   Generate new data: sample z ~ N(0,I) ──► Decoder ──► new x
```

### Process / functionality
1. **Encode** input to latent vector(s)
2. (VAE) sample `z` via reparameterisation
3. **Decode** to reconstruction
4. Compute loss, backprop through both nets jointly
5. **After training**: use the encoder for features/compression/anomaly scores; use the decoder (VAE) for generation/interpolation

### Use cases
Dimensionality reduction & visualisation · denoising (images, audio, ECG) · **anomaly detection** (high reconstruction error = anomaly) · pretraining/feature learning · compression · image inpainting · generative modelling (VAE, VQ-VAE, latent diffusion's first stage) · recommender systems · drug discovery (molecular latent spaces)

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---|---|
| Unsupervised, needs no labels | Reconstructions of VAEs often blurry |
| Learns compact features, nonlinear PCA | Plain AE may just learn identity if too big |
| Stable training (vs GANs) | Latent space not structured (plain AE) |
| VAE: principled probabilistic generator | Posterior collapse / β-tuning in VAEs |
| Great for anomaly detection | Can reconstruct anomalies too well |

### PyTorch code
```python
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- 1) Dense autoencoder ----------
class AutoEncoder(nn.Module):
    def __init__(self, in_dim=784, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                     nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(),
                                     nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, in_dim), nn.Sigmoid())
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

ae = AutoEncoder()
x = torch.rand(32, 784)
recon, z = ae(x)
loss = F.mse_loss(recon, x)

# --- Denoising: corrupt input, reconstruct the clean one ---
noisy = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)
loss_dae = F.mse_loss(ae(noisy)[0], x)

# --- Anomaly detection with reconstruction error ---
@torch.no_grad()
def anomaly_scores(model, x):
    model.eval()
    return ((model(x)[0] - x) ** 2).mean(dim=1)       # high score = anomaly
# threshold e.g. 99th percentile of scores on normal validation data

# ---------- 2) Convolutional autoencoder ----------
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),      # 28 -> 14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),     # 14 -> 7
            nn.Conv2d(32, 64, 7))                                      # 7 -> 1  (latent: 64)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), nn.ReLU(),                              # 1 -> 7
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 14
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())# 28
    def forward(self, x): return self.dec(self.enc(x))

print(ConvAE()(torch.rand(4, 1, 28, 28)).shape)                       # (4,1,28,28)

# ---------- 3) Variational autoencoder ----------
class VAE(nn.Module):
    def __init__(self, in_dim=784, hidden=400, latent=20):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.mu, self.logvar = nn.Linear(hidden, latent), nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))  # logits

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

def vae_loss(logits, x, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + beta * kl) / x.size(0)

vae = VAE()
x = torch.rand(32, 784)
logits, mu, logvar = vae(x)
print("VAE loss:", vae_loss(logits, x, mu, logvar).item())

# --- Generation & latent interpolation ---
@torch.no_grad()
def generate(vae, n=16):
    return torch.sigmoid(vae.dec(torch.randn(n, 20)))                 # sample prior -> decode
@torch.no_grad()
def interpolate(vae, x1, x2, steps=8):
    z1, z2 = vae.encode(x1)[0], vae.encode(x2)[0]
    return torch.stack([torch.sigmoid(vae.dec((1 - a) * z1 + a * z2)) for a in torch.linspace(0, 1, steps)])
```

---

## 8. Graph Neural Networks (GNN)

### Basics
GNNs operate on **graphs** `G = (V, E)`: nodes (entities) connected by edges (relations), with irregular structure that grids/sequences can't represent: social networks, molecules, road networks, knowledge graphs. A GNN learns node/edge/graph embeddings by letting nodes **exchange messages with their neighbours**.

### Key concepts
- **Node features `X ∈ ℝ^{N×F}`**, **adjacency matrix `A ∈ ℝ^{N×N}`** (or `edge_index` list), optional edge features
- **Message passing** (core paradigm), per layer:
  1. **Message** from neighbours  2. **Aggregate** (sum/mean/max/attention; must be permutation-invariant)  3. **Update** node state
  `h_v^{(k+1)} = UPDATE( h_v^{(k)}, AGGREGATE({ h_u^{(k)} : u ∈ N(v) }) )`
- **k layers = k-hop receptive field**
- **Over-smoothing**: too many layers make all node embeddings similar (use 2–4 layers, residuals, jumping knowledge)
- **Readout/pooling**: node embeddings → one graph vector (sum/mean/attention pooling)
- **Task levels**: node (classification), edge (link prediction), graph (property prediction)
- **Layer types**
  | Layer | Idea |
  |---|---|
  | **GCN** | Normalised neighbourhood average: `H' = σ(D̃^{-½} Ã D̃^{-½} H W)`, `Ã = A + I` |
  | **GraphSAGE** | Sample neighbours + aggregate (mean/LSTM/pool); inductive, scalable |
  | **GAT** | Attention weights per neighbour |
  | **GIN** | Sum aggregation + MLP; as expressive as WL test |
  | MPNN / Edge conv / Graph Transformers | General / geometric / global attention variants |
- **Transductive** (one fixed graph, e.g. Cora) vs **inductive** (new graphs at test time)

### Architecture
```
 Graph input                     Message passing (layer k → k+1)                 Task heads
                                                                              
     (B)───(C)            neighbours' features of node A                 Node-level:  h_v → Linear → class
    /  \     \                     B ─┐                                 Edge-level:  [h_u‖h_v] → MLP → link?
  (A)──(D)───(E)                   C ─┼─► AGGREGATE (mean/sum/attn)     Graph-level: READOUT(all h_v) → MLP
                                   D ─┘          │
  X: node features                               ▼
  A: connectivity           h_A^(k) ──► UPDATE( h_A^(k), agg ) ──► h_A^(k+1)

  Full model:
   X, edge_index ─► [GNN layer + ReLU + Dropout] ─► [GNN layer + ReLU] ─► [GNN layer] ─► embeddings
                         1-hop info                    2-hop info            3-hop info        │
                                                                                               ▼
                                                                          Pooling (graph tasks) ─► Classifier
```

### Process / functionality
1. Represent data as graph: node features, edges (`edge_index` of shape `[2, E]`)
2. Each GNN layer: gather neighbour features → aggregate → transform → update
3. Stack layers to widen receptive field
4. Task head on node/edge/graph embeddings → loss (e.g. cross-entropy on labelled nodes only for semi-supervised learning)
5. Mini-batching: large graphs via neighbour sampling (`NeighborLoader`); many small graphs via block-diagonal batching (`DataLoader`)

### Use cases
Node classification (paper topics, fraud accounts) · link prediction & recommendation (Pinterest PinSage) · molecule property/drug discovery · traffic/ETA prediction (Google Maps) · knowledge graph completion · protein structure · particle physics · social-network analysis · combinatorial optimisation · fake news detection

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---|---|
| Handles irregular, relational data natively | Over-smoothing / over-squashing with depth |
| Permutation invariant, inductive biases | Scalability on huge graphs (neighbourhood explosion) |
| Few parameters, uses structure + features | Limited expressiveness (WL bound) |
| Works at node, edge and graph level | Graph construction is often non-trivial |
| Flexible: heterogeneous/dynamic graphs | Less standardised tooling than CV/NLP |

### PyTorch code: pure PyTorch (no extra library, to show the mechanics)
```python
import torch, torch.nn as nn, torch.nn.functional as F

class GCNLayer(nn.Module):
    """H' = act( D^-1/2 (A+I) D^-1/2 · H · W )  — dense version for small graphs."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_f))
    def forward(self, x, adj):
        a = adj + torch.eye(adj.size(0), device=adj.device)     # add self-loops
        d_inv_sqrt = a.sum(1).pow(-0.5)
        a_norm = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]  # symmetric normalisation
        return a_norm @ self.lin(x) + self.bias

class GCN(nn.Module):
    def __init__(self, in_f, hid, n_cls, p=0.5):
        super().__init__()
        self.l1, self.l2, self.p = GCNLayer(in_f, hid), GCNLayer(hid, n_cls), p
    def forward(self, x, adj):
        x = F.dropout(F.relu(self.l1(x, adj)), self.p, self.training)
        return self.l2(x, adj)

# Toy graph: 5 nodes, 4 features, 3 classes
adj = torch.tensor([[0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0]], dtype=torch.float)
x = torch.randn(5, 4)
print(GCN(4, 8, 3)(x, adj).shape)                                # (5, 3)

# --- Sparse message passing with edge_index (scales to big graphs) ---
def mean_aggregate(x, edge_index):
    src, dst = edge_index                                        # edges src -> dst
    out = torch.zeros_like(x).index_add_(0, dst, x[src])         # sum messages per target node
    deg = torch.zeros(x.size(0), device=x.device).index_add_(0, dst, torch.ones(src.size(0), device=x.device))
    return out / deg.clamp(min=1).unsqueeze(-1)
```

### PyTorch Geometric (PyG): practical usage (`pip install torch_geometric`)
```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch_geometric.loader import DataLoader

# ================= Node classification (Cora citation network) =================
dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]                       # x:[2708,1433], edge_index:[2,10556], y, train/val/test masks

class NodeGNN(nn.Module):
    def __init__(self, in_c, hid, out_c, kind="gcn", heads=8):
        super().__init__()
        if kind == "gcn":
            self.c1, self.c2 = GCNConv(in_c, hid), GCNConv(hid, out_c)
        elif kind == "sage":
            self.c1, self.c2 = SAGEConv(in_c, hid), SAGEConv(hid, out_c)
        elif kind == "gat":
            self.c1 = GATConv(in_c, hid, heads=heads, dropout=0.6)
            self.c2 = GATConv(hid * heads, out_c, heads=1, dropout=0.6)
    def forward(self, x, edge_index):
        x = F.dropout(x, 0.5, self.training)
        x = F.elu(self.c1(x, edge_index))
        x = F.dropout(x, 0.5, self.training)
        return self.c2(x, edge_index)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NodeGNN(dataset.num_features, 16, dataset.num_classes, "gcn").to(device)
data = data.to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train(); opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])  # labelled nodes only
    loss.backward(); opt.step()
model.eval()
pred = model(data.x, data.edge_index).argmax(1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f"test acc: {acc:.3f}")      # ≈ 0.80-0.82 for GCN on Cora

# ================= Graph classification (e.g. molecules, MUTAG) =================
ds = TUDataset(root="data/TUDataset", name="MUTAG")
loader = DataLoader(ds.shuffle(), batch_size=32)

class GraphGNN(nn.Module):
    def __init__(self, in_c, hid, n_cls):
        super().__init__()
        mlp = lambda i, o: nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))
        self.c1, self.c2 = GINConv(mlp(in_c, hid)), GINConv(mlp(hid, hid))
        self.head = nn.Linear(hid, n_cls)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.c1(x, edge_index))
        x = F.relu(self.c2(x, edge_index))
        return self.head(global_mean_pool(x, batch))           # readout: one vector per graph

gmodel = GraphGNN(ds.num_features, 64, ds.num_classes)
for g in loader:
    logits = gmodel(g.x, g.edge_index, g.batch); break

# ================= Link prediction idea =================
# z = encoder(x, edge_index)                      # node embeddings
# score = (z[src] * z[dst]).sum(-1)               # dot-product decoder
# loss = F.binary_cross_entropy_with_logits(score, labels)   # + negative sampling
```

---

## 9. Comparison Matrix & Architecture Chooser

### Side-by-side
| Architecture | Data type | Key inductive bias | Parallel training | Long-range context | Typical params | Main weakness |
|---|---|---|---|---|---|---|
| **MLP** | Vectors / tabular | None (fully connected) | ✅ | n/a | small | No structure awareness |
| **CNN** | Grids (images, audio) | Locality, translation equivariance | ✅ | Limited (depth) | small–medium | Global context, rotation |
| **RNN** | Sequences | Sequential order, shared weights | ❌ | Poor | small | Vanishing gradients |
| **LSTM/GRU** | Sequences | Gated memory | ❌ | Medium | medium | Slow, limited length |
| **Transformer** | Sequences / sets / patches | Attention (few biases) | ✅ | Excellent | medium–huge | O(n²), data hungry |
| **GAN** | Generates any (mostly images) | Adversarial game | ✅ | depends on G/D | medium | Unstable training |
| **Autoencoder/VAE** | Any | Bottleneck compression | ✅ | depends | small–medium | Blurry (VAE), identity risk |
| **GNN** | Graphs | Permutation-invariant message passing | ✅ (batched) | Limited (k hops) | small | Over-smoothing, scaling |

### Decision guide
```
What is your data?
├─ Tabular / flat vectors ............ MLP (also try gradient boosting)
├─ Images / spectrograms / grids ..... CNN (or ViT with lots of data / pretrained)
├─ Text / language ................... Transformer (pretrained: BERT, GPT, T5...)
├─ Time series ....................... LSTM/GRU or 1D-CNN (small data); Transformer (large data)
├─ Graphs / molecules / networks ..... GNN (GCN → GraphSAGE/GAT/GIN)
├─ Need to GENERATE new samples
│    ├─ sharp images, fast sampling .. GAN (or diffusion for best quality)
│    └─ smooth latent space, density . VAE
└─ No labels / anomaly / compression . Autoencoder
```

---

## 10. Practical Cheat Sheet

### Tensor shape conventions
| Model | Input shape | Notes |
|---|---|---|
| MLP | `(B, F)` | flatten with `x.flatten(1)` |
| CNN (2D) | `(B, C, H, W)` | channels-first |
| 1D CNN | `(B, C, L)` | `Conv1d` expects channels first |
| RNN/LSTM | `(B, T, F)` with `batch_first=True` | default without it is `(T, B, F)` |
| Transformer | `(B, T, D)` | with `batch_first=True` |
| GNN (PyG) | `x:(N,F)`, `edge_index:(2,E)` | batch vector for graph-level tasks |

### Common PyTorch snippets
```python
# Count parameters
sum(p.numel() for p in model.parameters() if p.requires_grad)

# Save / load
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location=device)); model.eval()

# Mixed precision (faster on GPU)
scaler = torch.amp.GradScaler("cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(x), y)
scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

# LR schedulers
torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=1000)
torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)

# Early stopping (minimal)
best, wait = float("inf"), 0
# if val < best: best, wait = val, 0; torch.save(...)  else: wait += 1; stop if wait > patience

# Reproducibility
torch.manual_seed(42); torch.cuda.manual_seed_all(42)

# Compile for speed (PyTorch 2.x)
model = torch.compile(model)

# Gradient accumulation (simulate bigger batch)
# loss = loss / accum_steps; loss.backward(); if (i+1) % accum_steps == 0: opt.step(); opt.zero_grad()

# Freeze / unfreeze layers
for p in model.backbone.parameters(): p.requires_grad = False
```

### Debugging checklist
- ☐ Overfit a **tiny batch** first; the loss should go → 0
- ☐ Right loss for the head? (`CrossEntropyLoss` takes **logits**, not softmax output)
- ☐ `model.train()` / `model.eval()` set correctly?
- ☐ `optimizer.zero_grad()` called every step?
- ☐ Inputs normalised? Labels dtype `long` for classification?
- ☐ Loss `NaN`? → lower LR, clip gradients, check for `log(0)`/division by zero
- ☐ Loss not decreasing? → LR too low/high, check data pipeline & label alignment
- ☐ Train ≫ val performance → more regularisation/augmentation/data
- ☐ Shapes: print `x.shape` after every layer once

### Hyperparameter starting points
| Setting | MLP | CNN | LSTM | Transformer | GAN | GNN |
|---|---|---|---|---|---|---|
| Optimizer | Adam | AdamW/SGD+mom. | Adam/RMSprop | AdamW | Adam(β₁=0.5) | Adam |
| LR | 1e-3 | 1e-3 (SGD 0.1) | 1e-3 | 1e-4–3e-4 + warmup | 2e-4 | 1e-2–1e-3 |
| Batch | 32–256 | 32–256 | 32–128 | 16–512 | 64–128 | full-graph / 32 graphs |
| Regularise | Dropout 0.2–0.5 | BN + aug | Dropout 0.2, clip 1.0 | Dropout 0.1, wd 0.01 | label smooth, SN | Dropout 0.5, wd 5e-4 |
| Depth | 2–4 | 5–50 (ResNet) | 1–3 | 6–96 | 3–5 each | 2–3 layers |

### Further reading
- *Deep Learning* (Goodfellow, Bengio, Courville) · *Dive into Deep Learning* (d2l.ai) · PyTorch docs & tutorials (pytorch.org)
- Papers: *Attention Is All You Need* (2017) · *GAN* (2014) · *VAE* (2013) · *ResNet* (2015) · *LSTM* (1997) · *GCN* (2016) · *GAT* (2017)
- Libraries: `torchvision`, `torchaudio`, `transformers`, `torch_geometric`, `pytorch-lightning`, `timm`

---
*End of reference card.*
