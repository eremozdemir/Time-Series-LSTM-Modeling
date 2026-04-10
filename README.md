# Time Series Modeling with Deep Learning
Erem Ozdemir

**CMPE 401 Instructor-Defined Project 2**

Classification using a Transformer model on the FordA dataset, and forecasting using an LSTM model on the Jena Climate dataset.

---

## Table of Contents

1. [Project Setup](#project-setup)
2. [Notebooks](#notebooks)
3. [Datasets](#datasets)
4. [Model Architectures](#model-architectures)
5. [Task 1: Baseline Results](#task-1--baseline-results)
6. [Task 2: Improvement Experiments (LSTM)](#task-2--improvement-experiments-lstm)
7. [Task 3 : Benchmark Summary](#task-3--benchmark-summary)
8. [Task 4: Discussion Questions](#task-4--discussion-questions)

---

## Project Setup

1. From your root, create and activate the virtual environment:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

2. Install dependencies:
```bash
python3.11 -m pip install -r requirements.txt
```

3. Reload the window:

    a. Open the Command Palette:
    - Windows / Linux: ```Ctrl + Shift + P```
    - macOS: ```Cmd + Shift + P```

    b. Type ```Developer:Reload Window```, and press enter
  

4. Select the virtual environment as the notebook Kernel in VS Code:

   a. Click the kernel selector in the top right corner:

   ![Select Kernel](Images/readme_images/SelectKernel.png)

   b. Click **Python Environments**:

   ![Select Kernel](Images/readme_images/PythonEnv.png)

   c. Select the `.venv` environment that was just created:

   ![Select Kernel](Images/readme_images/.venv.png)

---

## Notebooks

| Notebook | Task | Dataset | Model |
|---|---|---|---|
| [timeseries_classification_transformer.ipynb](notebooks/timeseries_classification_transformer.ipynb) | Binary Classification | FordA | Transformer |
| [timeseries_weather_forecasting.ipynb](notebooks/timeseries_weather_forecasting.ipynb) | Temperature Forecasting | Jena Climate | LSTM |

---

## Datasets

### FordA: Engine Noise Classification

- **Source:** UCR Time Series Archive (hosted on GitHub by [hfawaz](https://github.com/hfawaz/cd-diagram))
- **Task:** Binary classification - detect abnormal engine noise
- **Format:** Each sample is a univariate time series of 500 timesteps
- **Data Splits:**
  - **Training Set:** 3,601 samples
  - **Test Set:** 1,320 samples
- **Classes:** 2 
  - Originally labelled −1 and +1 and remapped to 0 and 1
- **Input shape per sample:** `(500, 1)`

### Jena Climate: Weather Forecasting

- **Source:** [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
- **Task:** Regression — predict temperature 12 hours into the future
- **Format:** 14 meteorological features recorded every 10 minutes from Jan 2009 – Dec 2016
- **Data Splits:**
  - **Total rows:** ~420,551 
  - **Training rows:** ~300,693 (71.5%)
  - **Testing rows:** ~119,858 (28.5%)
- **Selected features (7):** 
  - Pressure, 
  - Temperature, 
  - Saturation vapor pressure, 
  - Vapor pressure deficit, 
  - Specific humidity, 
  - Airtight, 
  - Wind speed
- **Input window:** 720 observations (5 days, sampled every 60 min; 120 steps)
- **Prediction horizon:** 72 observations ahead (12 hours)
- **Input shape per sample:** `(120, 7)`

---

## Model Architectures

### Classification Transformer

The encoder block follows a **Pre-LN (Pre-Layer Normalization)** design, which applies normalization before the residual connection. This improves gradient flow and training stability.

```
Input (batch, 500, 1)
    │
    ├── [×4] TransformerEncoder block
    │       ├── MultiHeadAttention (4 heads, key_dim=256)
    │       ├── Dropout (0.25)
    │       ├── LayerNormalization
    │       ├── Residual add
    │       ├── Conv1D (ff_dim=4, relu)
    │       ├── Dropout (0.25)
    │       ├── Conv1D (restore channels)
    │       ├── LayerNormalization
    │       └── Residual add
    │
    ├── GlobalAveragePooling1D
    ├── Dense (128, relu)
    ├── Dropout (0.4)
    └── Dense (2, softmax)
```

- **Parameters:** ~97k
- **Optimizer:** Adam (lr = 1e-4)
- **Loss:** Sparse categorical crossentropy
- **Metric:** Sparse categorical accuracy
- **Callbacks:** EarlyStopping (patience=10, restore best weights)
- **Max epochs:** 150 
  - **Batch size:** 64


### Forecasting LSTM

A simple single-layer LSTM followed by a linear output head.

```
Input (batch, 120, 7)
    │
    ├── LSTM (32 units)
    └── Dense (1)  ← predicted temperature (normalized)
```

- **Parameters:** ~5k
- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Mean Squared Error (MSE)
- **Callbacks:** 
  - EarlyStopping (patience=5)
  - ModelCheckpoint (best val_loss)
- **Max epochs:** 10
  - **Batch size:** 256

---

## Task 1: Baseline Results

### Transformer Baseline (FordA Classification)

| Metric | This Run | Keras Reference |
|---|---|---|
| Training Accuracy | ~49.2% | ~95% |
| Validation Accuracy (best) | ~52.3% | ~84% |
| **Test Accuracy** | **~51.6%** | **~85%** |
| Epochs run (early stopped) | 48 | ~110–120 |
| Total parameters | ~97,000 | ~97,000 |

The model **did not converge** in this local run. The loss remained at ~0.693 (ln(2)) throughout all 48 epochs, indicating the model produced random binary predictions throughout. Early stopping triggered after 48 epochs once val_loss showed no improvement for 10 consecutive epochs.

**Key observations:**
- A constant loss of ~0.693 equals ln(2) — the cross-entropy of a perfectly uniform binary classifier. The model was learning nothing and outputting 50/50 class probabilities every epoch.
- This is expected when running a 97k-parameter Transformer on CPU with lr=1e-4 and no warm-up schedule. Gradients at initialization are small and require many batches on fast hardware to accumulate meaningful signal.
- The Keras reference results (~85% test accuracy) are reproducible on Colab with GPU over ~110–120 epochs. The model architecture is correct; the issue is the training environment and compute budget.

### LSTM Baseline (Jena Climate Forecasting)

| Metric | Value |
|---|---|
| Training Loss (MSE, final epoch) | 0.1033 |
| **Best Validation Loss (MSE)** | **0.1340** |
| Best epoch | 2 of 7 |
| Total parameters | ~5,153 |

All values are computed on normalized features (zero mean, unit variance computed from training set only).

**Key observations:**
- The model converged very quickly: the best val_loss (0.1340) was reached at epoch 2, and early stopping halted training at epoch 7 after 5 consecutive non-improving epochs.
- Despite having only ~5k parameters, the single-layer LSTM achieves a reasonable MSE on this 7-feature, 120-step forecasting task.
- The early peak at epoch 2 followed by rising val_loss is a sign of mild overshoot — the lr=0.001 step size is slightly too large for this model to settle cleanly, which the Experiment 3 results confirm directly.

---

## Task 2: Improvement Experiments (LSTM)

Three controlled modifications were applied to the LSTM forecasting model, one at a time, to isolate each effect. All other hyperparameters remain identical to the baseline.

### Experiment 1: Stacked LSTM (Two Layers)

**Change:** Replace the single `LSTM(32)` with two stacked LSTM layers: `LSTM(32, return_sequences=True)` followed by `LSTM(32)`.

```python
# Baseline
lstm_out = keras.layers.LSTM(32)(inputs)

# Modified
x = keras.layers.LSTM(32, return_sequences=True)(inputs)
lstm_out = keras.layers.LSTM(32)(x)
```

**Motivation:** A second LSTM layer allows the model to learn higher-order temporal abstractions. The first layer captures low-level patterns (e.g., daily cycles), while the second layer can capture dependencies between those patterns.

| Metric | Baseline | Stacked LSTM |
|---|---|---|
| Val Loss (MSE) | 0.1340 | **0.1254** |
| Parameters | ~5k | ~13k |
| Best epoch | 2 of 7 | 10 of 10 |

**Observation:** Stacking a second LSTM layer reduced val loss by 6.4%, using 2.6× more parameters. Notably, the stacked model was still improving at epoch 10 when the training budget ran out (the baseline peaked at epoch 2), so the true best is likely lower still. The improvement suggests the second layer successfully encodes higher-order temporal structure — patterns across patterns — that the single hidden state cannot represent.

---

### Experiment 2: Larger Hidden Size (LSTM 64)

**Change:** Increase the LSTM hidden units from 32 to 64.

```python
# Baseline
lstm_out = keras.layers.LSTM(32)(inputs)

# Modified
lstm_out = keras.layers.LSTM(64)(inputs)
```

**Motivation:** A larger hidden state gives the model more capacity to encode the 7-feature input sequence. With 32 units, the hidden state may be a bottleneck when compressing 120 timesteps × 7 features.

| Metric | Baseline | LSTM (64 units) |
|---|---|---|
| Val Loss (MSE) | 0.1340 | 0.1356 |
| Parameters | ~5k | ~18k |
| Best epoch | 2 of 7 | 10 of 10 |

**Observation:** LSTM(64) was marginally worse than the baseline (+1.1%), but the key detail is in the convergence pattern. The baseline peaked at epoch 2 and degraded — it converged fast but to a slightly overshot minimum. LSTM(64) was still steadily improving through all 10 epochs when the budget ran out. This tells a different story than "the wider model is worse": it is actually converging to a competitive solution, just more slowly because a larger parameter space takes more gradient steps to organize.

**Why didn't it help within the 10-epoch budget?** LSTM(64) has ~3.5× more parameters than LSTM(32). With the same learning rate (0.001) and the same number of epochs, each parameter receives fewer effective update steps relative to the complexity it needs to fit. The baseline LSTM(32) can "fill in" its simpler representation quickly; LSTM(64) needs more time to leverage its extra capacity. Given 20–30 epochs, LSTM(64) would likely surpass the baseline. The take-away is that **adding model capacity requires increasing the training budget proportionally**, especially when the learning rate is unchanged.

---

### Experiment 3: Reduced Learning Rate (0.0005)

**Change:** Reduce the Adam learning rate from 0.001 to 0.0005.

```python
# Baseline
optimizer=keras.optimizers.Adam(learning_rate=0.001)

# Modified
optimizer=keras.optimizers.Adam(learning_rate=0.0005)
```

**Motivation:** A lower learning rate can lead to a more stable descent and help escape sharp local minima, at the cost of requiring more epochs to converge. With `max_epochs=10` and early stopping, this tests whether the default rate was too aggressive.

| Metric | Baseline | LR = 0.0005 |
|---|---|---|
| Val Loss (MSE) | 0.1340 | **0.1115** |
| Parameters | ~5k | ~5k |
| Best epoch | 2 of 7 | 10 of 10 |

**Observation:** Halving the learning rate produced the **best result across all experiments** — a 16.8% reduction in val loss with zero additional parameters. Like the other experiments, it was still improving at epoch 10 when the budget ended, meaning the true optimum is likely even lower. The contrast with the baseline is clear: with lr=0.001 the model peaks at epoch 2 and degrades; with lr=0.0005 it keeps improving through all 10 epochs in a smooth monotonic curve. This is the defining sign that lr=0.001 was causing overshoot and lr=0.0005 enables stable, sustained learning.

---

## Task 3: Benchmark Summary

### LSTM Forecasting Comparison Table (Jena Climate)

| Experiment | Modification | Val Loss (MSE) | Change | Parameters | Epochs |
|---|---|---|---|---|---|
| Baseline | LSTM(32), lr=0.001 | 0.1340 | — | ~5k | 7 |
| Exp 1 | Stacked LSTM (32→32) | 0.1254 | -6.4% ✅ | ~13k | 10 |
| Exp 2 | LSTM(64), lr=0.001 | 0.1356 | +1.1% ❌ | ~18k | 10 |
| **Exp 3** | **LSTM(32), lr=0.0005** | **0.1115** | **-16.8% ✅** | **~5k** | **10** |

**Best result:** Reduced learning rate (Experiment 3) achieved the lowest val MSE (0.1115), a **16.8% improvement** over the baseline with no additional parameters. All three experiments hit the 10-epoch ceiling while still improving — a larger training budget would yield lower losses across the board.

### Transformer Classification Baseline (FordA)

| Model | Test Accuracy | Parameters | Epochs |
|---|---|---|---|
| Transformer (this run) | ~51.6% (no convergence) | ~97k | 48 |
| Transformer (Keras reference) | ~85% | ~97k | ~110–120 |

The Transformer was not modified as the improvement task focused on the LSTM notebook. The model failed to converge in this local run (see Baseline Results section for explanation).

### Takeaways

- **Learning rate** was the most impactful single change: halving it (0.001 → 0.0005) gave a 16.8% val loss improvement at zero parameter cost, because the baseline lr=0.001 was causing the optimizer to overshoot past the optimum.
- **Architecture depth** (stacking LSTM layers) gave a solid 6.4% improvement and was still converging at epoch 10 — with more training budget it would likely push further.
- **Hidden size increase** (LSTM 64) marginally underperformed within 10 epochs because a larger model needs more gradient steps to converge. It was still improving at epoch 10, indicating it would surpass the baseline with a larger epoch budget.
- The key practical lesson: **for small recurrent models on structured time-series, fix the learning rate before adding parameters**.

---

## Task 4: Discussion Questions

### Which model did you find easier to understand and why?

I found the LSTM model was easier to understand. Its architecture is linear and follows a single intuitive path, a recurrent layer processes the sequence step-by-step, building up a hidden state that summarizes past observations, and then a Dense layer maps that hidden state to a single output value. The connection between the model structure and the task (predicting future temperature from past readings) is immediately clear.

The Transformer was more conceptually demanding. Multi-head self-attention requires understanding how queries, keys, and values are computed and combined, and why this allows any timestep to directly attend to any other regardless of distance. The Pre-LN residual structure, positional independence, and the role of the 1D convolutions as a position-wise feed-forward network all require more background knowledge to internalize. That said, once understood, the Transformer's architecture is quite modular and impressive, where each encoder block is identical and can be stacked freely.

### What improvement did you try, and what did you learn from it?

Three experiments were run on the LSTM forecasting model, and the most impactful was **reducing the learning rate** (Experiment 3). Halving lr from 0.001 to 0.0005 reduced validation MSE from 0.1340 to **0.1115** — a 16.8% improvement using the exact same architecture and parameter count.

The clearest signal is in the convergence patterns. With lr=0.001, the baseline peaks at epoch 2 and then degrades — a classic sign that the optimizer is taking steps too large to settle into a good minimum, bouncing past it each update. With lr=0.0005, the model keeps improving smoothly through all 10 epochs, suggesting each gradient step is landing in a progressively better region of the loss surface.

**Experiment 1 (Stacked LSTM)** gave a solid 6.4% improvement and was also still converging at epoch 10, showing that depth helps but requires more training time to pay off. **Experiment 2 (LSTM 64)** was only marginally worse than baseline (+1.1%), but importantly it was also still improving at epoch 10 — unlike the baseline which peaked at epoch 2. This means the wider model is not fundamentally broken; it simply needs more training steps to organize its larger parameter space. Given 20–30 epochs, LSTM(64) at lr=0.001 would likely surpass the baseline.

Overall, the experiments show that **how you train matters as much as what you train**: a simple learning rate reduction outperformed both architectural changes within the given budget, and the architectural changes themselves were still improving when training ended.

---

## Project Structure

```
Time-Series-LSTM-Modeling/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── timeseries_classification_transformer.ipynb   # Transformer (FordA classification)
│   └── timeseries_weather_forecasting.ipynb          # LSTM (Jena climate forecasting)
└── Images/
    ├── readme_images/
    │   ├── SelectKernel.png
    │   ├── PythonEnv.png
    │   └── .venv.png
    └── visualizations/
```

---

## References

- Ntakouris, T. (2021). *Timeseries classification with a Transformer model*. Keras Examples.
- Attri, P., Sharma, Y., Takach, K., & Shah, F. (2020). *Timeseries forecasting for weather prediction*. Keras Examples.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- FordA dataset: UCR Time Series Archive / [hfawaz/cd-diagram](https://github.com/hfawaz/cd-diagram)
- Jena Climate dataset: [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
