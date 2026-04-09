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
python3 -m venv .venv && source .venv/bin/activate
```

2. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

3. Select the virtual environment as the notebook Kernel in VS Code:

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
| [timeseries_classification_transformer.ipynb](timeseries_classification_transformer.ipynb) | Binary Classification | FordA | Transformer |
| [timeseries_weather_forecasting.ipynb](timeseries_weather_forecasting.ipynb) | Temperature Forecasting | Jena Climate | LSTM |

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

| Metric | Value |
|---|---|
| Training Accuracy | ~95% |
| Validation Accuracy | ~84% |
| **Test Accuracy** | **~85%** |
| Epochs to converge | ~110–120 |
| Total parameters | ~97,000 |

The model converges reliably without hyperparameter tuning. Early stopping prevents overfitting; restoring best weights ensures the reported test accuracy reflects the optimal checkpoint.

**Key observations:**
- The model learns quickly in the first 30 epochs, then refines more gradually.
- The gap between training (~95%) and validation (~84%) accuracy indicates mild overfitting, which is expected given the relatively small dataset size (~3,600 samples) and the model's capacity (~97k parameters).
- The Transformer's self-attention mechanism allows each timestep to attend to all others globally, which suits the FordA classification task since the discriminative signal may not be locally contiguous.

### LSTM Baseline (Jena Climate Forecasting)

| Metric | Value |
|---|---|
| Training Loss (MSE) | ~0.0579 |
| **Validation Loss (MSE)** | **~0.0628** |
| Epochs to converge | ~6–8 (early stopping) |
| Total parameters | ~5,000 |

All values are computed on normalized features (zero mean, unit variance computed from training set only).

**Key observations:**
- Despite having only ~5k parameters, the single-layer LSTM achieves a reasonable MSE on this 7-feature, 120-step sequence task.
- Training converges quickly (within 10 epochs), suggesting the problem is not particularly hard for this horizon (12 hours) and that a small hidden state is sufficient to capture the dominant temperature dynamics.
- The prediction plots show accurate tracking of the general trend, with some lag at sharp temperature transitions. This is a known limitation of fixed-horizon single-step LSTM forecasters.

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
| Val Loss (MSE) | ~0.0628 | ~0.0571 |
| Parameters | ~5k | ~13k |
| Epochs | ~7 | ~8 |

**Observation:** Stacking LSTM layers reduced validation MSE by ~9%, at the cost of ~2.6× more parameters and slightly longer training. The improvement suggests that the second layer successfully captures inter-cycle dependencies that a single layer cannot.

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
| Val Loss (MSE) | ~0.0628 | ~0.0592 |
| Parameters | ~5k | ~18k |
| Epochs | ~7 | ~7 |

**Observation:** Doubling the hidden size modestly improved validation MSE (~6% reduction) with no change in convergence speed. This suggests that the hidden state size was a mild bottleneck in the baseline.

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
| Val Loss (MSE) | ~0.0628 | ~0.0614 |
| Parameters | ~5k | ~5k |
| Epochs | ~7 | ~9 |

**Observation:** The lower learning rate required more epochs to converge but produced slightly better final validation loss (~2% improvement). This indicates the default learning rate was mildly overshooting near the optimum. The improvement is marginal, suggesting that learning rate is not the primary bottleneck for this model.

---

## Task 3: Benchmark Summary

### LSTM Forecasting Comparison Table (Jena Climate)

| Experiment | Modification | Val Loss (MSE) | Parameters | Epochs |
|---|---|---|---|---|
| Baseline | LSTM(32), lr=0.001 | ~0.0628 | ~5k | ~7 |
| Exp 1 | Stacked LSTM (32→32) | **~0.0571** | ~13k | ~8 |
| Exp 2 | LSTM(64), lr=0.001 | ~0.0592 | ~18k | ~7 |
| Exp 3 | LSTM(32), lr=0.0005 | ~0.0614 | ~5k | ~9 |

**Best result:** Stacked LSTM (Experiment 1) achieved the lowest validation MSE (~0.0571), a ~9% improvement over the baseline.

### Transformer Classification Baseline (FordA)

| Model | Test Accuracy | Parameters | Epochs |
|---|---|---|---|
| Transformer (baseline) | ~85% | ~97k | ~115 |

The Transformer was not modified as the improvement task focused on the LSTM notebook.

### Takeaways

- **Architecture depth** (stacking layers) was the most impactful change for the LSTM model.
- **Hidden size** provided a smaller but consistent gain.
- **Learning rate reduction** offered marginal benefit given the constrained 10-epoch training budget.
- The Transformer achieves strong accuracy (~85%) on a short univariate sequence with no feature engineering, demonstrating that global self-attention is effective even for 1D time-series classification.

---

## Task 4: Discussion Questions

### Which model did you find easier to understand and why?

I found the LSTM model was easier to understand. Its architecture is linear and follows a single intuitive path, a recurrent layer processes the sequence step-by-step, building up a hidden state that summarizes past observations, and then a Dense layer maps that hidden state to a single output value. The connection between the model structure and the task (predicting future temperature from past readings) is immediately clear.

The Transformer was more conceptually demanding. Multi-head self-attention requires understanding how queries, keys, and values are computed and combined, and why this allows any timestep to directly attend to any other regardless of distance. The Pre-LN residual structure, positional independence, and the role of the 1D convolutions as a position-wise feed-forward network all require more background knowledge to internalize. That said, once understood, the Transformer's architecture is quite modular and impressive, where each encoder block is identical and can be stacked freely.

### What improvement did you try, and what did you learn from it?

The primary improvement was **stacking two LSTM layers** (Experiment 1). This modification reduced validation MSE from ~0.0628 to ~0.0571 (~9% improvement).

The key insight was that a single recurrent layer must compress the entire temporal pattern into one hidden state transition per step, which can be a bottleneck when the input sequence contains multiple levels of structure (e.g., short-term hourly variation and longer multi-day trends). Adding a second LSTM layer allows the first layer to produce a richer intermediate representation (passed along via `return_sequences=True`), and the second layer to summarize higher-order patterns across those intermediate features.

The experiment also showed that deeper is not always better in direct proportion to parameter count as Experiment 2 used more parameters (18k vs 13k for the stacked model) but achieved a smaller improvement. This illustrates that **how capacity is structured** (depth vs. width) matters as much as the total number of parameters.

---

## Project Structure

```
Time-Series-LSTM-Modeling/
├── README.md
├── requirements.txt
├── timeseries_classification_transformer.ipynb   # Transformer (FordA classification)
├── timeseries_weather_forecasting.ipynb          # LSTM (Jena climate forecasting)
└── Images/
    ├── SelectKernel.png
    ├── PythonEnv.png
    └── .venv.png
```

---

## References

- Ntakouris, T. (2021). *Timeseries classification with a Transformer model*. Keras Examples.
- Attri, P., Sharma, Y., Takach, K., & Shah, F. (2020). *Timeseries forecasting for weather prediction*. Keras Examples.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- FordA dataset: UCR Time Series Archive / [hfawaz/cd-diagram](https://github.com/hfawaz/cd-diagram)
- Jena Climate dataset: [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
