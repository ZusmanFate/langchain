""" 
本文件是【部署一个鲜花网络电商的人脉工具（上）】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388071003886190632
您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
"""
# 设置OpenAI API密钥
import os

os.environ["SERPAPI_API_KEY"] = (
    "32c2066b72804a61e0b3d69b8dd05d3fd50b36063c6573d4ee16580222c857a7"
)

# 导入所取的库
import re
from agents.weibo_agent import lookup_V
from tools.general_tool import remove_non_chinese_fields
from tools.scraping_tool import get_data

if __name__ == "__main__":

    # 拿到UID
    response_UID = lookup_V(flower_type="牡丹")

    # 抽取UID里面的数字
    UID = re.findall(r"\d+", response_UID)[0]
    print("这位鲜花大V的微博ID是", UID)

    # 根据UID爬取大V信息
    person_info = get_data(UID)
    print(person_info)

    # 移除无用的信息
    remove_non_chinese_fields(person_info)
    print(person_info)
