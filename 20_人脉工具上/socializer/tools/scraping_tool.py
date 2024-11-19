import json
import requests
import time


# 定义爬取微博用户信息的函数
def scrape_weibo(url: str):
    """爬取相关鲜花服务商的资料"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
        "Referer": "https://weibo.com",
    }
    cookies = {"cookie": "PC_TOKEN=c06d216e0a; XSRF-TOKEN=4K2BcRDjqh9Gvcnqi25hMqoM; SCF=AruFk6ytrF1VQS-o64MiiaYnzY_tWyaHt0sE7c4f43x6jDp1DZgPGQFuhm7cnbsRXVayyIxlMvCmctL24oGywN8.; SUB=_2A25KPrnGDeRhGeNL6FMZ9izIyD-IHXVpNbMOrDV8PUNbmtANLUPEkW9NSQKE2jT7o2oKzB79sLpwE-7yzm-f_EZZ; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFx7ausaDnzZsLM6o5w-swT5NHD95QfSKep1hqEShe0Ws4DqcjKUsyLds80U2SyUJ8jIgHE1K.Nentt; ALF=02_1734497942; WBPSESS=_scRfOFisRo4_wxUfEnJIFec4xe6x88lnDNzdLeOjNx1ey_F7jJOQuPscsr_52eOAPFAhOGTgTI-Q9sdLS7i-nJQL-xWMBnHktVJMFUekhICH6cw2_imqRXgrF083MaNyZ9SWZoFEd2AqB1RfWAxIw==; _s_tentry=weibo.com; Apache=7321319379196.411.1731905974453; SINAGLOBAL=7321319379196.411.1731905974453; ULV=1731905974491:1:1:1:7321319379196.411.1731905974453:"}
    response = requests.get(url, headers=headers, cookies=cookies)
    time.sleep(3)  # 加上3s 的延时防止被反爬
    return response.text


# 根据UID构建URL爬取信息
def get_data(id):
    url = "https://weibo.com/ajax/profile/detail?uid={}".format(id)
    html = scrape_weibo(url)
    response = json.loads(html)

    return response
