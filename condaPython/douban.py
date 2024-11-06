from bs4 import BeautifulSoup
import requests
import urllib.request

header = {
    'Cookie':'bid=0BPoYVYn-NE; _pk_id.100001.4cf6=9b075407a0d5a87c.1721117053.; _pk_ses.100001.4cf6=1; ap_v=0,6.0; __utma=30149280.370578693.1721117053.1721117053.1721117053.1; __utmc=30149280; __utmz=30149280.1721117053.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utma=223695111.1564435602.1721117053.1721117053.1721117053.1; __utmb=223695111.0.10.1721117053; __utmc=223695111; __utmz=223695111.1721117053.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __yadk_uid=QxckU85KUxJPuwJ5XVRtMcW4JMQv98Bi; __utmb=30149280.6.8.1721117061074; RT=s=1721118845125&r=https%3A%2F%2Fmovie.douban.com%2Ftop250%3Fstart%3D0',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}

url = 'https://movie.douban.com/top250?start=0'

resp = requests.get(url,headers=header)
resp.text.encode('utf-8')
soup = BeautifulSoup(resp.text, 'lxml')
print(soup.prettify())