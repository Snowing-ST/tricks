# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:43:01 2020
银行授信额度统计

@author: situ
"""

import pandas as pd
import numpy as np
import os
import re

path = "F:/finance"    
encoding="gb18030"
file_names = os.listdir(path)
file_names = [f for f in file_names if len(re.findall(r"各银行融资情况",f))>0] #只读取后缀名为csv的文件
all_df = pd.DataFrame()
for i in range(len(file_names)):
#    try:
    temp_df = pd.read_excel(os.path.join(path,file_names[i]))
#    except:

#    temp_df = pd.read_csv(os.path.join(path,file_names[i]),encoding=encoding,engine = "python")

    all_df = pd.concat([all_df,temp_df],axis=0,ignore_index=True)
#        print(all_df.head())


# A集团18-20
all_df1 = pd.pivot_table(all_df[all_df["集团名称"]=="A集团"][["银行名称","实际额度","用信总额","时间"]],index=["银行名称"],columns=["时间"],values=["实际额度","用信总额"],aggfunc=[np.sum],fill_value=0,margins=0)
all_df1.columns=all_df1.columns.droplevel()
writer = pd.ExcelWriter(os.path.join(path,"A集团.xlsx"), engine='xlsxwriter')
 
all_df1.to_excel(writer,sheet_name = "sheet1",encoding = "gbk")
writer.save()   


#AB集团2020
all_df2 = pd.pivot_table(all_df[all_df["时间"]==2020][["银行名称","实际额度","用信总额"]],index=["银行名称"],values=["实际额度","用信总额"],aggfunc=[np.sum],fill_value=0,margins=0)
all_df2.columns=all_df2.columns.droplevel()
all_df2.sort_index(level=2,ascending=True)

writer2 = pd.ExcelWriter(os.path.join(path,"AB集团.xlsx"), engine='xlsxwriter')
all_df2.to_excel(writer2,sheet_name = "sheet1",encoding = "gbk")
writer2.save()   

# B集团18-20
all_df3 = pd.pivot_table(all_df[all_df["集团名称"]=="B集团"][["银行名称","实际额度","用信总额","时间"]],index=["银行名称"],columns=["时间"],values=["实际额度","用信总额"],aggfunc=[np.sum],fill_value=0,margins=0)
all_df3.columns=all_df3.columns.droplevel()
writer = pd.ExcelWriter(os.path.join(path,"B集团18-20.xlsx"), engine='xlsxwriter')
 
all_df3.to_excel(writer,sheet_name = "sheet1",encoding = "gbk")
writer.save()   

#AB集团18-20
all_df4 = pd.pivot_table(all_df[["银行名称","实际额度","用信总额","时间"]],index=["银行名称"],columns=["时间"],values=["实际额度","用信总额"],aggfunc=[np.sum],fill_value=0,margins=0)
all_df4.columns=all_df4.columns.droplevel()
all_df4.sort_index(level=2,ascending=True)

writer2 = pd.ExcelWriter(os.path.join(path,"AB集团18-20.xlsx"), engine='xlsxwriter')
all_df4.to_excel(writer2,sheet_name = "sheet1",encoding = "gbk")
writer2.save()   