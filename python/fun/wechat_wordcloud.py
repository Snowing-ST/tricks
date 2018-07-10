# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 17:58:12 2017

@author: HYPC
"""

import sys
import itchat
import wordcloud
reload(sys)

sys.getdefaultencoding()
sys.setdefaultencoding('utf8')  

itchat.login()
#爬取自己好友相关信息， 返回一个json文件
friends = itchat.get_friends(update=True)[0:]

#初始化计数器
male = female = other = 0
#friends[0]是自己的信息，所以要从friends[1]开始
for i in friends[1:]:
    sex = i["Sex"]
    if sex == 1:
        male += 1
    elif sex == 2:
        female += 1
    else:
        other +=1
#计算朋友总数
total = len(friends[1:])
#打印出自己的好友性别比例
print("男性好友： %.2f%%" % (float(male)/total*100) + "\n" +
"女性好友： %.2f%%" % (float(female) / total * 100) + "\n" +
"不明性别好友： %.2f%%" % (float(other) / total * 100))

def get_var(var):
    variable = []
    for i in friends:
        value = i[var]
        variable.append(value)
    return variable
#调用函数得到各变量，并把数据存到csv文件中，保存到桌面
NickName = get_var("NickName")
Sex = get_var('Sex')
Province = get_var('Province')
City = get_var('City')
Signature = get_var('Signature')
from pandas import DataFrame
data = {'NickName': NickName, 'Sex': Sex, 'Province': Province,
        'City': City, 'Signature': Signature}
frame = DataFrame(data)
frame.to_excel('C:/Users/situ.st.1/Desktop/SituXueying/data.xlsx', index=True)

import re
siglist = []
for i in friends:
    signature = i["Signature"].strip().replace("span","").replace("class","").replace("emoji","")
    rep = re.compile("1f\d+\w*|[<>/=]")
    signature = rep.sub("", signature)
    siglist.append(signature)
text = " ".join(siglist)

import jieba
wordlist = jieba.cut(text, cut_all=True)
word_space_split = " ".join(wordlist)


import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
import PIL.Image as Image
import os
coloring = np.array(Image.open("C:/Users/situ.st.1/Pictures/butterfly.jpg"))

my_wordcloud = WordCloud(background_color="white", max_words=2000,
                         mask=coloring, max_font_size=60, random_state=42, scale=2,
                         font_path=os.environ.get("FONT_PATH", "C:/Windows/Fonts/simfang.ttf"))
my_wordcloud.generate(word_space_split)
image_colors = ImageColorGenerator(coloring)
plt.figure(figsize=(30,25)) 
plt.imshow(my_wordcloud.recolor(color_func=image_colors))
plt.xticks([]),plt.yticks([]) #隐藏坐标线 
plt.axis("off")
plt.imshow(my_wordcloud)
plt.savefig("C:/Users/situ.st.1/Pictures/wordcloud.jpg")
plt.show()


