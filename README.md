# hackathon_zucchetti
Repository of the hackathon sponsored by Zucchetti done on 4th of April 2025


## Splitting train/test in time-series
Splitting time series data for training and testing machine learning models requires careful consideration because the temporal order of observations matters. Unlike standard cross-validation where you can randomly shuffle data, in time series, you must preserve the chronological order to avoid data leakage (using future information to predict the past/present).