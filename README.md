<center><h1>AtmoSeer</h1></center>

## Overview
AtmoSeer is a comprehensive full-stack data science project that brings predictive atmospheric science to your fingertips. Unlike most existing tools that only analyze historical greenhouse gas data, AtmoSeer leverages an advanced deep learning algorithm to forecast future emission levels with statistical confidence intervals.

This project represents a complete data pipeline: from extracting raw measurements directly from NOAA's Global Monitoring Laboratory and NASA MODIS databases, through advanced data engineering, to deploying state-of-the-art time series analysis models. AtmoSeer tracks four of the most potent greenhouse gases that drive climate change:

* Carbon Dioxide (CO₂)
* Methane (CH₄)
* Nitrous Oxide (N₂O)
* Sulfur Hexafluoride (SF₆)

At the core of AtmoSeer is a custom-built Bidirectional LSTM neural network architecture with an attention mechanism, optimized through Bayesian hyperparameter tuning. This model captures both long-term trends and seasonal patterns in atmospheric gas concentrations through cyclical seasonal awareness created during feature engineering, which pairs perfectly with the LSTM's ability to learn both long-term trends going back decades and recent patterns from only a few weeks prior.

AtmoSeer was designed as an open-source contribution to climate science, providing researchers, educators, and concerned citizens with powerful tools to understand and anticipate atmospheric changes that shape Earth's future.