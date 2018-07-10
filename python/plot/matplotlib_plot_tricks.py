#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:12:07 2018

@author: situ
"""

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
plt.style.use('ggplot')

tips = sns.load_dataset("tips")
sns.barplot(x="day", y="total_bill", data=tips)
plt.xlabel("hahahaha") #更改x轴标签
plt.xticks([1,2,3,4],["A","B","C","D"]) #更改x轴刻度
#如果x轴的刻度并不是连续型变量的取值，而是定性变量的取值，则一定是1 2 3 4

#加legend，线条
red_patch = mpatches.Patch(color='red', label="chargeoff")
blue_patch = mpatches.Patch(color='blue', label="fullypay")
plt.legend(handles=[red_patch,blue_patch])

#加legend，色块
red_line= mlines.Line2D([], [], color='red', label="chargeoff")
blue_line = mlines.Line2D([], [], color='blue', label="fullypaid")
plt.legend(handles=[red_line,blue_line])



#corrplot with seaborn
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(15,15))
sns.heatmap(data.corr(),annot=False, vmax=1,vmin=-1, square=True, cmap="RdBu")
