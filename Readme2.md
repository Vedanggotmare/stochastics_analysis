About stochastics_analysis

stochastics_analysis is a quantitative research project exploring methods for volatility modeling and time-series analysis in irregularly sampled financial datasets. It represents a compilation of experiments, models, and simulations that I’ve conducted to probe the intersection of stochastic analysis and financial time series.

Motivation & Objectives

Real-world financial data is often irregularly sampled (missing days, asynchronous ticks, etc.), which poses challenges for traditional time-series models that assume uniform sampling.

The goal is to build a hybrid modeling framework that can robustly handle irregular sampling and still extract meaningful volatility structure.

Specifically:
 1. Use Lomb–Scargle periodograms (a technique from astronomy) to detect periodic signals in irregular data.
 2. Combine the extracted periodic components with GARCH(1,1) modeling to forecast volatility more reliably than naive interpolation-based approaches.

Through this approach, I aim to investigate whether this combination improves volatility forecasting and model robustness, especially under data sparsity or irregular sampling patterns.

Key Features & Components

R scripts / modules (e.g. GARCH.R) implementing time-series workflows using packages such as quantmod, lomb, rugarch, tseries, and xts.

Python / Jupyter notebooks for simulation, data processing, and visualization (e.g. stochastic_process_analysis.ipynb).

Simulated experiments to validate methods:
 - Removing random subsets of timestamps to simulate missing data
 - Applying Lomb–Scargle to the “gappy” series
 - Fitting GARCH on the processed returns
 - Visualizing volatility clustering, periodic signals, and model fit

Visualization assets (e.g. Figure_1.png, Figure_2.png, etc.) to illustrate results and insights.

Workflow Overview

Fetch stock / financial data (e.g. via quantmod).

Introduce irregularity by randomly removing data points (to simulate real-world sampling challenges).

Apply the Lomb–Scargle periodogram to detect cycles or periodic components in the irregular series.

Compute log-returns and fit a GARCH(1,1) model using the preprocessed data.

Analyze how well the model captures volatility behavior and compare it against baseline methods (e.g. interpolation + GARCH).

Visualize and interpret the results.

Why This Matters

Traditional volatility models often rely on regularly spaced data; when faced with missing or irregularly spaced samples, they may misestimate volatility or exhibit bias.

By leveraging signal-processing techniques (like Lomb–Scargle) before volatility modeling, this project explores whether we can recover hidden structure and improve forecasts, especially in data‐sparse regimes.

It contributes to bridging methods from astrophysics / signal processing with quantitative finance / stochastic time-series modeling.
