#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:02:31 2018

@author: situ
"""

import os
import pandas as pd
import csv

os.chdir("/Users/situ/Downloads")


file = open("2015.csv") 
newfile = open("2015top100.csv","w")
writer = csv.writer(newfile)

for i in range(100000):
    line = file.readline()
    writer.writerow(line.strip().split(","))
#    print(line.strip().split(","))
file.close()
newfile.close()

