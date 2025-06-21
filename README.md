# Numerical Optimization with Python - Portfolio Optimization Project

## Authors

- Eldad Uzman
- Tom Ron
- Liron Zarhay

## Overview

This project implements advanced portfolio optimization techniques using numerical optimization methods in Python. The focus is on Conditional Value-at-Risk (CVaR) optimization and Mean-Variance optimization for financial portfolio management, with additional features for risk attribution and walk-forward analysis.

## Features

- **CVaR Optimization**: Implements Rockafellar & Uryasev's linear CVaR minimization approach
- **Mean-Variance Optimization**: Traditional Markowitz portfolio optimization
- **Risk Attribution**: Shapley-style risk contribution analysis for portfolio components
- **Walk-Forward Analysis**: Time-series backtesting with rolling optimization windows
- **Top-N Momentum Strategy**: Simple momentum-based portfolio strategy for comparison
- **Interactive Visualizations**: Plotly-based charts for risk attribution and performance analysis
- **Performance Profiling**: Built-in profiling tools for optimization performance analysis

## Core Algorithms

### CVaR Optimization

The project implements Rockafellar & Uryasev's linear formulation for CVaR minimization:

```python
def optimise_cvar(R, beta=0.95, target_return=None, short_cap=None):
    """
    Rockafellar-&-Uryasev linear CVaR minimisation.
    
    Parameters:
    - R: Returns DataFrame
    - beta: Confidence level (0, 1)
    - target_return: Optional return constraint
    - short_cap: Optional short selling constraint
    """
```

### Mean-Variance Optimization

Traditional Markowitz portfolio optimization:

```python
def optimize_mean_variance(R, target_return=None, short_cap=None):
    """
    Mean-variance optimisation with optional constraints.
    """
```

### Risk Attribution

Shapley-style risk contribution analysis:

```python
def cvar_risk_contributions(R, weights, beta=0.95):
    """Shapley-equivalent attribution for portfolio CVaR."""
    
def variance_risk_contributions(cov, weights):
    """Shapley (Euler) attribution for portfolio variance."""
```

## Data Format

The project expects stock price data in CSV format with:
- Date column (automatically detected)
- Price data in wide format (one column per stock)
- Or long format with columns: date, company, price

Example data structure:
```csv
date,MSFT,GOOGL,AAPL,...
2015-02-01,40.12,520.45,120.34,...
2015-02-02,40.89,525.12,121.67,...
...
```

## Research Background

This project is based on academic research in portfolio optimization, particularly:
- Rockafellar & Uryasev (2000): CVaR optimization
- Markowitz (1952): Mean-variance optimization
- Shapley value theory for risk attribution