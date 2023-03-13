import requests, random, json, os, csv, pandas as pd
from dotenv import load_dotenv


def import_data():
  load_dotenv('./.env')

  api_keys = os.getenv('API_KEYS')
  api_keys = api_keys.split(',')
  key_usage = {key: 0 for key in api_keys}

  df_data = pd.read_csv('./data/nasdaq_symbols.csv')

  remove = []

  for i, tckr in enumerate(df_data['Symbol']):

    remove.append(i)

    try:
      available_keys = [key for key in api_keys if key_usage[key] < 5]
      api_key = random.choice(available_keys)
      key_usage[api_key] += 1
      
      url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={tckr}&apikey={api_key}'
      response = requests.get(url)
      try:
        bal_sheet = response.json()['quarterlyReports']
        print("Balance Sheet Found")
      except KeyError:  
        print("skip")
        continue
      

      available_keys = [key for key in api_keys if key_usage[key] < 5]
      api_key = random.choice(available_keys)
      key_usage[api_key] += 1

      url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={tckr}&apikey={api_key}'
      response = requests.get(url)
      try:
        cash_flow = response.json()['quarterlyReports']
        print("Cash Flow Found")
      except KeyError:  
        print("skip")
        continue


      available_keys = [key for key in api_keys if key_usage[key] < 5]
      api_key = random.choice(available_keys)
      key_usage[api_key] += 1

      url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={tckr}&apikey={api_key}'
      response = requests.get(url)
      try:
        income_statement = response.json()['quarterlyReports']
        print("Income Statement Found")
      except KeyError:  
        print("skip")
        continue
        
    except:
      print('API limit reached')
      for index in sorted(remove, reverse=True):
        df_data = df_data.drop(index)

      df_data.to_csv('./data/nasdaq_symbols.csv', index=False)
      remove = []

      print("Going to sleep for 5 minutes")
      break

    for bal, cash, income in zip(bal_sheet, cash_flow, income_statement):
      quarter = {**{"ticker": tckr}, **bal, **cash, **income}

      with open('./data/company_statements.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(quarter.values())

# url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol={tckr}&apikey={api_key}'
# response = requests.get(url)
# data = response.json()
