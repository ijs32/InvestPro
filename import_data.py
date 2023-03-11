import requests, random, os, csv, pandas as pd
from dotenv import load_dotenv


def import_data():
  load_dotenv('./.env')

  api_keys = os.getenv('API_KEYS')
  api_keys = api_keys.split(',')

  df_data = pd.read_csv('./data/nasdaq_symbols.csv')

  remove = []

  for i, tckr in enumerate(df_data['Symbol']):
    
    api_key = random.choice(api_keys)

    try: 
      url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={tckr}&apikey={api_key}'
      response = requests.get(url)
      if response != {}:
        print(response.json())
        bal_sheet = response.json()['quarterlyReports']
      
      url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={tckr}&apikey={api_key}'
      response = requests.get(url)
      if response != {}:
        print(response.json())
        cash_flow = response.json()['quarterlyReports']

      url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={tckr}&apikey={api_key}'
      response = requests.get(url)
      if response != {}:
        print(response.json())
        income_statement = response.json()['quarterlyReports']

      remove.append(i)

      for quarter in {**{"ticker": tckr}, **bal_sheet, **cash_flow, **income_statement}:
        print(quarter)

        with open('./data/company_statements.csv', 'a', newline='') as csvfile:
          # create a writer object
          writer = csv.writer(csvfile)

          # write the data rows
          writer.writerow(quarter.values())
        
    except:
      print('API limit reached')
      for index in sorted(remove, reverse=True):
        df_data = df_data.drop(index)

      df_data.to_csv('./data/nasdaq_symbols.csv', index=False)
      break

    break
# url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol={tckr}&apikey={api_key}'
# response = requests.get(url)
# data = response.json()
