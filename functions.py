import os, json, requests, random, pandas as pd
from bs4 import BeautifulSoup
import datetime as dt
import yfinance as yf
from dateutil.relativedelta import relativedelta
from MySQLConn.mySQLdb import engine
from sqlalchemy import text

def get_10k(headers):
  """
  Make requests to SEC to get company 10ks
  """
  with os.scandir("./data/filing-history") as dir:
    for file in dir:
        
        try:
            if "submissions" not in file.name and file.is_file(): # submissions files seem to be for ETFs and other non-company filings
                with open(file.path) as f:
                    data = json.load(f)
                    if "Nasdaq" not in data["exchanges"] and "NYSE" not in data["exchanges"]: continue # dont want OTC stocks
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
                                insert = text("INSERT INTO company_10k_statements (file_name, ticker, exchange, report_date) VALUES (:file_name, :tckr, :exchange, :report_date)")
                                
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


def insert_sp500_performance():
    """
    Insert SP500 performance data into DB
    """
    get_10k_reportDates = text("SELECT DISTINCT report_date FROM company_10k_statements")
    with engine.connect() as conn:
        result = conn.execute(get_10k_reportDates)

    report_dates = [row[0] for row in result]
    df = pd.read_csv("./data/sp500_performance.csv")    
    for index, row in df.iterrows():
        date = row["Date"].split("-")
        df.at[index, 'Date'] = dt.date(int(date[0]), int(date[1]), int(date[2]))

    # get rounded report_dates
    # for index, row in df.iterrows():
    rounded_dates = dict()
    for report_date in report_dates:
        cloz_dict = {
            abs(report_date - row["Date"]) : row["Date"]
            for _, row in df.iterrows()}
        
        res = cloz_dict[min(cloz_dict.keys())]
        rounded_dates[report_date] = res
    
    for report_date, rounded_date in rounded_dates.items():
        start_value = df.loc[df["Date"] == rounded_date]["Value"].values[0]
        if len(df.loc[df["Date"] == rounded_date + relativedelta(years=1)]["Value"].values) < 1:
            end_value = 3951.39 # current s&p 500, for more recent 10k reports
            end_date = dt.date(2023, 3, 31)
        else:
            end_value = df.loc[df["Date"] == rounded_date + relativedelta(years=1)]["Value"].values[0]
            end_date = rounded_date + relativedelta(years=1)
        
        performance = round((((end_value - start_value) / start_value) * 100), 4)

        insert = text("INSERT INTO sp500_performance (report_date, rounded_start_date, rounded_end_date, start_value, end_value, performance_over_range) VALUES(:report_date, :rounded_date, :end_date, :start_value, :end_value, :performance)")

        params = {"report_date": report_date, "rounded_date": rounded_date, "end_date": end_date, "start_value": start_value, "end_value": end_value, "performance": performance} 
        with engine.connect() as conn:
            result = conn.execute(insert, params)
            conn.commit()
            if result.rowcount < 1:
                raise Exception("Insert failed")
            else:
                print("Row inserted successfully")


def insert_tckr_performance():
    query_statements = text("SELECT * FROM company_10k_statements")

    with engine.connect() as conn:
        result = conn.execute(query_statements)

    # row[0] = statement_id, [1] = file_name, [2] = ticker, [3] = exchange, [4] = report_date, [5] = rounded_report_date, [6] = rounded_eoy_date, [7] = yoy_performance, [8] = predicted_sentiment, [9] = performance_id
    for row in result.all():
        try: 
            start = dt.datetime.combine(row[5], dt.time())
            end = dt.datetime.combine(row[6], dt.time())
            
            print("start: ", start)
            print("end: ", end)
            print("statement_id: ", row[0])
            tckr = yf.Ticker(row[2])
            # fast access to subset of stock info (opportunistic)
            result = tckr.history(period="1y", start=start, end=end)
            start_value = result.iloc[0]["Close"]
            end_value = result.iloc[-1]["Close"]

            performance = round((((end_value - start_value) / start_value) * 100), 4)
            print(performance)

            insert = text("UPDATE company_10k_statements SET yoy_performance = :performance WHERE statement_id = :statement_id")
            params = {"performance": performance, "statement_id": row[0]}

            with engine.connect() as conn:
                result = conn.execute(insert, params)
                conn.commit()
        except Exception as e:
            print("Error: ", e)
            continue


insert_tckr_performance()