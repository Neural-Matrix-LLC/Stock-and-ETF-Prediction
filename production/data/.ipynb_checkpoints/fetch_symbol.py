import csv

stock_path = "stocks_and_etfs/stocks.csv"
etf_path = "stocks_and_etfs/etf.csv"

with open('some.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)