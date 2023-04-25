import os, json, requests, random, re, nltk, sys
import pandas as pd, numpy as np, gzip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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

                                with open(f"../data/raw-data/raw-company-statements-v2/{tckr}--{report_date}__{cik}--{acc_number}--{file_name}", "w") as f:
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
    df = pd.read_csv("../data/training-data/sp500_performance.csv")    
    for index, row in df.iterrows():
        date = row["Date"].split("-")
        df.at[index, 'Date'] = dt.date(int(date[0]), int(date[1]), int(date[2]))

    # get rounded report_dates
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
            if len(df.loc[df["Date"] == rounded_date + relativedelta(years=1, days=1)]["Value"].values) < 1:
                end_value = 3951.39 # current s&p 500, for more recent 10k reports
                end_date = dt.date(2023, 3, 31)
            else: # leap year screwed up our metrics
                print("==== LEAP YEAR ====")
                end_value = df.loc[df["Date"] == rounded_date + relativedelta(years=1, days=1)]["Value"].values[0]
                end_date = rounded_date + relativedelta(years=1, days=1)
        else:
            end_value = df.loc[df["Date"] == rounded_date + relativedelta(years=1)]["Value"].values[0]
            end_date = rounded_date + relativedelta(years=1)
        
        performance = round((((end_value - start_value) / start_value) * 100), 4)

        insert = text("INSERT INTO sp500_performance (report_date, rounded_start_date, rounded_end_date, start_value, end_value, sp500_performance) VALUES(:report_date, :rounded_date, :end_date, :start_value, :end_value, :performance)")

        params = {"report_date": report_date, "rounded_date": rounded_date, "end_date": end_date, "start_value": start_value, "end_value": end_value, "performance": performance} 
        with engine.connect() as conn:
            result = conn.execute(insert, params)
            conn.commit()
            if result.rowcount < 1:
                raise Exception("Insert failed")
            else:
                print("Row inserted successfully")


def insert_tckr_performance():
    """
    SELECTS * FROM company_10k_statements and assigns them year performance in percentage
    """
    query_statements = text("SELECT * FROM company_10k_statements WHERE company_performance IS NULL")

    with engine.connect() as conn:
        result = conn.execute(query_statements)

    # row[0] = statement_id, [1] = file_name, [2] = ticker, [3] = exchange, [4] = report_date, [5] = rounded_report_date, [6] = rounded_eoy_date, [7] = company_performance, [8] = predicted_sentiment, [9] = performance_id
    for row in result.all():

        try: 
            start = dt.datetime.combine(row[5], dt.time())
            end = dt.datetime.combine(row[6], dt.time())
            
            print("statement_id: ", row[0])
            tckr = yf.Ticker(row[2])
            # fast access to subset of stock info (opportunistic)
            result = tckr.history(period="1y", start=start, end=end)

            start_value = result.iloc[0]["Close"]
            end_value = result.iloc[-1]["Close"]

            print(f"start: {start}")
            print(f"end: {end}")

            performance = round((((end_value - start_value) / start_value) * 100), 4)
            print(performance)

            insert = text("UPDATE company_10k_statements SET company_performance = :performance WHERE statement_id = :statement_id")
            params = {"performance": performance, "statement_id": row[0]}

            with engine.connect() as conn:
                result = conn.execute(insert, params)
                conn.commit()
        except Exception as e:
            print(f"Error: {e} for ticker: {row[2]}")
            continue


def calculate_sentiment_score():
    query = text("""
        SELECT cs.statement_id, cs.company_performance, sp.sp500_performance
            FROM company_10k_statements `cs`
        LEFT JOIN sp500_performance `sp` ON sp.performance_id = cs.performance_id
            WHERE cs.company_performance IS NOT NULL
        """)

    with engine.connect() as conn:
        result = conn.execute(query)

    for row in result:
        percent_above_below_SPY = round(((row[1] - row[2]) / 100), 2)
        sentiment_score = ((percent_above_below_SPY + 1) / 2)
        statement_id = row[0]
        
        insert = text("""
            UPDATE company_10k_statements `cs` 
                SET 
                cs.percent_above_below_SPY = :percent_above_below_SPY,
                cs.sentiment_score = :sentiment_score
            WHERE cs.statement_id = :statement_id
            """)
        
        params = {
            "percent_above_below_SPY": percent_above_below_SPY, 
            "sentiment_score": sentiment_score,
            "statement_id": statement_id
        }
        
        with engine.connect() as conn:
            result = conn.execute(insert, params)
            conn.commit()
            if result.rowcount < 1:
                raise Exception("Insert failed")
            else:
                print("Row inserted successfully")


def clean_data():
    # nltk.download('stopwords')
    # nltk.download('punkt')
    file_count = len(os.listdir("../data/raw-data/raw-company-statements-v2"))
    filter_count = 0

    with os.scandir("../data/raw-data/raw-company-statements-v2") as dir:
        for file in dir:
            try:
                labels = file.name.split("--")
                query = text("SELECT statement_id, sentiment_label FROM company_10k_statements WHERE file_name = :file_name AND ticker = :ticker LIMIT 1")
                params = {"file_name": labels[-1], "ticker": labels[0]}
                with engine.connect() as conn:
                    result = conn.execute(query, params)
                
                for row in result:
                    statement_id, sentiment_label = row
                    print(statement_id)
                    print(sentiment_label)
                    
                    with open(file.path) as f:
                        document = f.read()

                        document = re.sub(r'[^a-zA-Z0-9\s]', '', document)
                        document = re.sub(r'\b\w*\d\w*\b', '', document)

                        document = re.sub(r'\s{2,}', ' ', document)
                        document = document.lower()

                        stop_words = set(stopwords.words('english'))
                        words = word_tokenize(document)
                        filtered_words = [word for word in words if word.lower() not in stop_words]
                        document = ' '.join(filtered_words)
                        with open(f"../data/training-data/company-statements/{sentiment_label}__{statement_id}__{labels[0]}.txt", "w") as f:
                            f.write(document)
                    filter_count += 1
                    print(f"Progress: {filter_count}/{file_count} Files cleaned -- {((filter_count / file_count) * 100):.2f}% done.", end="\r")
            except Exception as e:
                if e is KeyboardInterrupt:
                    sys.exit()

                print("\n")
                print("Company removed from dataset")
                os.remove(file.path)
                continue

def apply_labels():
    with os.scandir("../data/training-data/company-statements") as dir:
        for file in dir:
            try:
                id = file.name.split("__")[0]
                query = text("SELECT sentiment_score FROM company_10k_statements WHERE statement_id = :statement_id LIMIT 1")

                params = {"statement_id": id}
                with engine.connect() as conn:
                    result = conn.execute(query, params)
                sentiment_label = result.first()[0]

                os.rename(f"../data/training-data/company-statements/{file.name}", f"../data/training-data/company-statements/{sentiment_label}__{file.name}")
            except Exception as e:
                if e is KeyboardInterrupt:
                    sys.exit()
                
                print(f"failed")


def save_glove():
    # the following code was taken from:
    # https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a

    vocab,embeddings = [],[]
    with open('./data/GloVe/50d/glove.6B.50d.txt','rt') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
    
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')

    pad_emb_npa = np.zeros((1,embs_npa.shape[1])) 
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))

    print("embeddings: ", embs_npa)
    print("vocab: ", vocab_npa)

    with open('./data/GloVe/50d/vocab_npa.npy','wb') as f:
        np.save(f,vocab_npa)

    with open('./data/GloVe/50d/embs_npa.npy','wb') as f:
        np.save(f,embs_npa)


def fix_labels():
        with os.scandir("../data/training-data/company-statements") as dir:
            for file in dir:
                try:
                    file_items = file.name.split("__")
                    print(file.name)

                    os.rename(f"../data/training-data/company-statements/{file.name}", f"../data/training-data/company-statements/{file_items[1]}__{file_items[2]}")
                except:
                    print("whatever")


def compress_data():
    """compress training data to save memory"""
    with os.scandir("../data/training-data/company-statements") as dir:
        for file in dir:
            try:
                with open(file.path, 'rb') as f_in:
                    # open the output file and compress the data
                    with gzip.open(f'../data/training-data/company-statements_gz/{file.name}.gz', 'wb') as f_out:
                        f_out.writelines(f_in)
                    print(f"Compressed {file.name} successfully")
            except Exception as e:
                print(e)
                continue