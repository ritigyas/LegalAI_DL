import requests
from bs4 import BeautifulSoup

def search_indian_kanoon(query):
    url = f"https://indiankanoon.org/search/?formInput={query}"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    results = []
    for a in soup.select(".result_title a")[:5]:
        results.append(a.text.strip())

    return results