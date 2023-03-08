import requests, json, os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')

url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol=IBM&apikey={api_key}'
url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=IBM&apikey={api_key}'
url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol=IBM&apikey={api_key}'
url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=IBM&apikey={api_key}'
response = requests.get(url)
data = response.json()


print(json.dumps(data, indent=4, sort_keys=True))
