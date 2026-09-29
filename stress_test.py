import concurrent.futures
import urllib.request
import time

url = "http://127.0.0.1:8001/docs"

def request(_):
    try:
        urllib.request.urlopen(url, timeout=10)
        return True
    except:
        return False

start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    results = list(executor.map(request, range(100)))

print("Concurrent Requests:", len(results))
print("Successful:", sum(results))
print("Failed:", len(results) - sum(results))
print("Time:", round(time.time() - start, 2), "seconds")
