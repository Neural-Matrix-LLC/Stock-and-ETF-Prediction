# Stock-and-ETF-Prediction
Looking into a set of features that are based on market data to predict a stock is going to up or down stock price at least by +2% or -2% on next trading day.

Input features:
1. Daily price of a set of stock of OPEN, HIGH, LOW, CLOSE, (ADJ-CLOSE)
2. The key indexes: S&P 500, Nasdaq 100, Hang Seng Index,
3. may put additional features of technical indicators based on stock price

# Data
[Google Drive](https://drive.google.com/file/d/1KJWGDK7rCGNb97JzJn4odw2EgVlyS350/view?usp=sharing)

## Expected Output
Build Framework (enable future iterations/improvements): We need a process framework will be able to modify the features set and able to find best parameters of model based of the modified features set. A accurate measurement to ensure the model accuracy on the prediction. The final model will run against look forward data set to get the actual accuracy.

Accuracy based on whether the model is able to predict whether the stock price changes by a degree of 2%.

# Project Workflow
1. Data acquisition
2. Data preprocessing
3. Develop and implement model
4. Backtest model
5. Optimization

# Data Preprocessing
Problems with Financial data: scarcity, non-stationarity, and state-dependence. Financial data exhibits very low signal-to-noise ratio. Neural networks are low bias and high-variance learners: models will overfit the noise in the data.

# Prerequisites
```
pip install -r requirements.txt
jupyter labextension install jupyterlab-plotly
```
