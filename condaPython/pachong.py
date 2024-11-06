# encodings='utf-8'
from bs4 import BeautifulSoup
import requests
import re

header = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}
url = 'http://www.cntour.cn/'
strhtml = requests.get(url, headers=header)
soup = BeautifulSoup(strhtml.text, 'lxml')
data = soup.select('#photoListABUIABACGAAg9uX_oQYo8bvQ3wMw0AU4ywQ > a > img')

for item in data:
    result = {
        'title': item['alt'],
        'link': item['data-original']
    }
    print(result)
