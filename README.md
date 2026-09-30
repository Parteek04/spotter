# Freight Rate Prediction — Machine Learning Engineer Assessment

## Overview

This repository contains my solution for the Spotter Machine Learning Engineer assessment.

The objective is to predict the **posted freight rate** for unseen loads using the supplied historical development dataset and generate predictions for the 12,000 validation loads.

The solution focuses on:

- Data exploration and quality checks
- Time-aware validation
- Feature engineering
- Categorical and numerical feature handling
- Missing-value handling
- CatBoost regression
- Multi-seed model ensembling
- Reproducible prediction generation
- December 2025 rate prediction and visualization

---

## Problem

For each freight load, the goal is to estimate its expected `posted_rate` using information such as:

- Pickup location
- Delivery location
- Distance
- Equipment type
- Weight
- Pickup/delivery coordinates
- Market index
- Quote signal
- Date

The final model is used to predict rates for all 12,000 unseen loads in:

`validation.csv`

The predictions are saved in:

`validation_predictions.csv`
