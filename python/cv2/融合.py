# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 16:43:03 2017

@author: situ.st.1
"""

import cv2
import numpy as np
import pandas as pd
#from __future__ import division
from matplotlib import pyplot as plt

img1 = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/7.2/5093F10D.jpg')
img2 = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/7.2/5093F10M.jpg')
img1.shape
img2.shape
#img1 = cv2.resize(img1,(384,288),interpolation=cv2.INTER_CUBIC)
img_mix = cv2.addWeighted(img1,0.5, img2, 0.5, 0)
img_mix1 = cv2.addWeighted(img1,1, img2, 0.5, 0)
cv2.imwrite("C:/Users/situ.st.1/Desktop/SituXueying/7.2/0505.jpg",img_mix)
cv2.imwrite("C:/Users/situ.st.1/Desktop/SituXueying/7.2/105.jpg",img_mix1)
#cv2.circle(img1,(122,129),2,(0,255,255),1)
#cv2.circle(img2,(91,111),3,(255,0,255),1)
cv2.imshow('img_mix', img_mix)
cv2.imshow('img_mix1', img_mix1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#按位运算
import cv2
import numpy as np
img1 = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/7.2/5093F10D.jpg')
img2 = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/7.2/5093F10M.jpg')
rows,cols,channel = img2.shape
roi = img1[0:rows, 0:cols ]



img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) #灰度
ret, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY) 
mask_inv = cv2.bitwise_not(mask) ##反色，即对二值图每个像素取反 
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv) #background                       
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask) #前景色，对原图像和掩膜进行位运算
#img2_fg = cv2.bitwise_and(roi,roi,mask = mask) 
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img_mix = cv2.addWeighted(img1_bg,0.5, img2_fg, 0.5, 0)


cv2.imshow('img1_bg',img1_bg)
cv2.imshow('img2_fg',img2_fg)
cv2.imshow('dst',dst)
cv2.imshow('img_mix', img_mix)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1 = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/7.2/5093F10D.jpg')
img2 = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/7.2/5093F10M.jpg')

if img1.size == img2.size and img1.dtype == img2.dtype:
     res = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
     cv2.namedWindow("show", cv2.WINDOW_NORMAL)
     cv2.imshow("show", res)
     if cv2.waitKey(0) == 27:
         cv2.destroyAllWindows()