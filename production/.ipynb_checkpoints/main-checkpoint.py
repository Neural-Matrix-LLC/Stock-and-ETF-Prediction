import logging
from data import fetch_data, processing_lstm

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
host="143.244.188.157"
port="3306"
user="patrick-finProj"
password="Pat#21$rick"


def main():
    logging.info(f'Start main.py')
    fetch_data.get_df()
    
    
    
    
    

# OUTPUT
# Date DMY
# symbol
# exchange
# 8 numbers