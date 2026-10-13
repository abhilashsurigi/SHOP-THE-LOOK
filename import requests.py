
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

product_id = "5a997eb7ea9e521e3c151423bd08e6df"

url = (
    f"https://i.pinimg.com/400x/"
    f"{product_id[:2]}/{product_id[2:4]}/"
    f"{product_id[4:6]}/{product_id}.jpg"
)

try:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(req, timeout=10) as response:
        print("Status:", response.status)
        print("Content type:", response.headers.get("Content-Type"))
        print("URL:", url)

except HTTPError as e:
    print("HTTP error:", e.code)
except URLError as e:
    print("Connection error:", e.reason)
except Exception as e:
    print("Error:", e)
