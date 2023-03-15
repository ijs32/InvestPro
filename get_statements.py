import os, json, requests, random, time
from bs4 import BeautifulSoup
from MySQLConn.mySQLdb import engine
from sqlalchemy import text

def get_10k(headers):
  """
  Make requests to SEC to get company 10ks
  """
  with os.scandir("./data/filing-history") as dir:
    for file in dir:
        
        try:
            if "submissions" not in file.name and file.is_file():
                with open(file.path) as f:
                    data = json.load(f)
                    if "Nasdaq" not in data["exchanges"] and "NYSE" not in data["exchanges"]: continue
                    if len(data["tickers"]) < 1: continue

                    tckr = data['tickers'][0]
                    cik = data['cik']
                    filings = data['filings']['recent']

                    limit = 0
                    for form, acc_number, file_name, report_date in zip(filings['form'], filings['accessionNumber'], filings["primaryDocument"], filings["reportDate"]):
                        if limit == 5: break
                        if form == '10-K':
                            if random.randrange(2) == 1:
                                acc_number = acc_number.replace("-", "")
                                report_date = report_date.replace("-", "")

                                response = requests.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_number}/{file_name}", headers=headers)
                                soup = BeautifulSoup(response.text, features="html.parser")
                                clean_text = soup.get_text()

                                with open(f"./data/raw-company-statements-v2/{tckr}--{report_date}__{cik}--{acc_number}--{file_name}", "w") as f:
                                    f.write(clean_text)
                                print(f"Downloaded {file_name} for {tckr}")

                                exchange = data["exchanges"][0]
                                insert = text("INSERT INTO company_10k_statements (fileName, ticker, exchange, reportDate) VALUES (:file_name, :tckr, :exchange, :report_date)")
                                
                                params={"file_name": file_name, "tckr": tckr, "exchange": exchange, "report_date": report_date}
                                with engine.connect() as conn:
                                    result = conn.execute(insert, params)
                                    conn.commit()
                                    if result.rowcount < 1:
                                        raise Exception("Insert failed")
                                print(f"Inserted {file_name} for {tckr} into DB")

                                limit += 1
            print(f"Removing: {file.path}")
            os.remove(file.path)
        # in some cases we request too frequently, this exception puts us on cool down.
        except Exception as e:
            print(f"Error: {e}")
            os.remove(file.path)
            return
        

if __name__ == "__main__":
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}

    while True:
        get_10k(headers)
        print("Going to sleep for 5 minutes...")
        time.sleep(300)