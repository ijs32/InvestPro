from functions import get_10k
import time

if __name__ == "__main__":
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}

    while True:
        get_10k(headers)
        print("Going to sleep for 5 minutes...")
        time.sleep(300)