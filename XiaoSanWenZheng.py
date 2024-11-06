import requests
from bs4 import BeautifulSoup

header = {
    'Cookie': 'PHPSESSID_bbs=d10bc6992b98c681870f240fe5f070cd; Hm_lvt_07d59984cbd1413f629a3c2f6769143a=1725937803; '
              'HMACCOUNT=60B459BBF9AF20C0; Hm_lpvt_07d59984cbd1413f629a3c2f6769143a=1725937840',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,'
              'application/signed-exchange;v=b3;q=0.7',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 '
                  'Safari/537.36 '
}
num = 100
for i in range(1, num+1):
    url = f'https://wz.xsnet.cn/bbs/thread/lists/2?flag=&keywords=&page={i}'
    strhtml = requests.get(url, headers=header)
    soup = BeautifulSoup(strhtml.text, 'lxml')
    data = soup.find_all('td',attrs={'class':'td-subject p-l-0'})
    # print(data[2])

    for item in data:
        zuzhi = item.find('span',attrs={'style':'background:#999;color:white;'}).text
        if zuzhi == '闻堰街道':
            print(url)
            user = item.find('a',attrs={'class':'username m-r-xs'}).text
            coin = item.find('a',attrs={'target':'_self'}).text
            data = item.find('span').text
            print(f"{zuzhi}:{data}:{user}")
            print(coin)

