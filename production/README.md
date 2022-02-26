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
│   ├── {symbol_date}.csv
├── logging
│   ├── app.log
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```
