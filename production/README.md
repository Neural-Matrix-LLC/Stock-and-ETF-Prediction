# Folder structure

```
├── data
│   ├── histdailyprice3
│   |   |   ├── {symbol}.csv
│   ├── stocks_and_etfs
│   │   ├── etf_list.csv
│   │   ├── stock_list.csv
│   ├── data.py
│   ├── processing.py
├── logging
│   ├── app.log
├── model
│   ├── params
│   |   ├── model={garch, svr, mlp}
│   |   |   ├── {symbol}.json
│   ├── lstm
│   |   ├── {symbol}}.h5
│   ├── garch.py
│   ├── svr.py
│   ├── mlp.py
│   ├── lstm.py
├── output
│   ├── dailyoutput_{today}.csv
├── predict_final
│   ├── predict_{today}.csv
├── main.py
├── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```
