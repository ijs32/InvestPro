import os, json, requests, random
from bs4 import BeautifulSoup

def get_10k(headers):
  """
  Make requests to SEC to get company 10ks
  """
  with os.scandir("./data/filing-history") as dir:
      
      for file in dir:
          
        if "submissions" not in file.name and file.is_file():
            with open(file.path) as f:
                data = json.load(f)
                if "Nasdaq" not in data["exchanges"]: continue
                if len(data["tickers"]) < 1: continue
                
                tckr = data['tickers'][0]
                cik = data['cik']
                filings = data['filings']['recent']

                limit = 0
                for form, acc_number, file_name, report_date in zip(filings['form'], filings['accessionNumber'], filings["primaryDocument"], filings["reportDate"]):
                    if limit == 5: break
                    if form == '10-K':
                        if random.randrange(3) == 1:
                          acc_number = acc_number.replace("-", "")
                          report_date = report_date.replace("-", "")

                          response = requests.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_number}/{file_name}", headers=headers)
                          soup = BeautifulSoup(response.text, features="html.parser")
                          clean_text = soup.get_text(strip=True)

                          with open(f"./data/company-statements_html/{tckr}__{report_date}__{file_name}", "w") as f:
                              f.write(clean_text)
                          print(f"Downloaded {file_name} for {tckr}")

                          limit += 1
        print(f"Removing: {file.path}")
        os.remove(file.path)


if __name__ == "__main__":
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}
    get_10k(headers)