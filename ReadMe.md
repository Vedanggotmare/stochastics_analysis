# stochastics\_analysis

**stochastics\_analysis** is a quantitative research project exploring methods for volatility modeling and time-series analysis in irregularly sampled financial datasets. It represents a compilation of experiments, models, and simulations that probe the intersection of stochastic analysis and financial time series.

---

## Motivation & Objectives

* Real-world financial data is often **irregularly sampled** (missing days, asynchronous ticks, etc.), which poses challenges for traditional time-series models that assume uniform sampling.
* The goal is to build a **hybrid modeling framework** that can robustly handle irregular sampling and still extract meaningful volatility structure.
* Specifically:

  1. Use **Lomb–Scargle periodograms** (a technique from astronomy) to detect periodic signals in irregular data.
  2. Combine the extracted periodic components with **GARCH(1,1)** modeling to forecast volatility more reliably than naive interpolation-based approaches.
* This project investigates whether this combination improves volatility forecasting and model robustness, especially under data sparsity or irregular sampling patterns.

---

## Key Features & Components

* **R scripts / modules** (e.g. `GARCH.R`) for time-series workflows using:
  `quantmod`, `lomb`, `rugarch`, `tseries`, and `xts`.
* **Python / Jupyter notebooks** for simulation, data processing, and visualization (e.g. `stochastic_process_analysis.ipynb`).
* **Simulated experiments** to validate methods:

  * Removing random subsets of timestamps to simulate missing data
  * Applying Lomb–Scargle to the “gappy” series
  * Fitting GARCH on the processed returns
  * Visualizing volatility clustering, periodic signals, and model fit
* **Visualization assets** (e.g. `Figure_1.png`, `Figure_2.png`) showcasing results.

---

## Workflow Overview

1. Fetch stock / financial data (e.g. via `quantmod`).
2. Introduce irregularity by randomly removing data points.
3. Apply the Lomb–Scargle periodogram to detect cycles or periodic components.
4. Compute log-returns and fit a GARCH(1,1) model using the preprocessed data.
5. Compare model performance against baseline methods (e.g. interpolation + GARCH).
6. Visualize and interpret the results.

---

## Why This Matters

* Traditional volatility models often rely on regularly spaced data; with irregular samples, they can misestimate volatility or exhibit bias.
* By leveraging **signal-processing techniques** (like Lomb–Scargle) before volatility modeling, this project explores whether we can **recover hidden structure** and improve forecasts.
* It bridges methods from astrophysics / signal processing with quantitative finance / stochastic time-series modeling.

---

## Repository Structure

```
stochastics_analysis/
├── R_scripts/                # R-based models and workflows
│   └── GARCH.R
├── notebooks/                # Python Jupyter notebooks
│   └── stochastic_process_analysis.ipynb
├── figures/                  # Visualization outputs
│   ├── Figure_1.png
│   ├── Figure_2.png
│   └── ...
└── README.md
```

---

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/Vedanggotmare/stochastics_analysis.git
   cd stochastics_analysis
   ```

2. Install dependencies:

   * **R**: `quantmod`, `lomb`, `rugarch`, `tseries`, `xts`
   * **Python**: `numpy`, `pandas`, `matplotlib`, `scipy`, `arch`, `jupyter`

3. Run the R scripts or open the Jupyter notebooks to reproduce results.
