# markov-portfolio-switcher

A regime-switching dynamic portfolio allocation system for Indian equity markets (Nifty 50).

## Overview

Combines Hidden Markov Model (HMM) regime detection with Markov chain transition forecasting to predict short-term regime probabilities, then allocates capital based on a forward-looking risk-adjusted signal.

## Pipeline

1. **Data** — 10 years of Nifty 50 daily OHLCV data (2014–2024)
2. **Regime Detection** — 4-state Gaussian HMM (Bull / Bear / HighVol / Crash)
3. **Transition Matrix** — MLE estimation from regime label sequence
4. **Regime Statistics** — Per-regime mean return and variance (with shrinkage for rare regimes)
5. **Forecasting** — Markov chain h-step forecast: μ_{t+h} = μ_t × P^h
6. **Signal** — Blended expected return / volatility (forward Sharpe signal)
7. **Backtest** — Cumulative return vs buy-and-hold benchmark

## Results (2014–2024, Nifty 50)

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Final Return | 4.17x | 3.35x |
| Sharpe Ratio | 1.09 | 0.80 |
| Max Drawdown | -35.35% | -40.04% |

## Tech Stack

Python, hmmlearn, numpy, pandas, matplotlib, scikit-learn

## Status

Work in progress — walk-forward validation (script 07) in progress.
