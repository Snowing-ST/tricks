# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 11:46:25 2017
特征匹配

@author: situ.st.1
"""

#Brute-Force 匹配  cv2.BFMatcher()

import numpy as np
import cv2
from matplotlib import pyplot as plt
#img1 = cv2.imread('C:/Users/situ.st.1/Pictures/box.png',0) # queryImage
#img2 = cv2.imread('C:/Users/situ.st.1/Pictures/box_in_scene.png',0) # trainImage
img1 = cv2.imread("C:/Users/situ.st.1/Desktop/SituXueying/8.4/EG/day-21/teeth_mask/t_1002a002.JPG",0)
img2 = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/8.4/EG/day-21_wl/teeth_mask/t_1002a001.JPG',0)
# Initiate SIFT detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)


# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
img3 = np.zeros((img2.shape[0]*2,img2.shape[1]*2),dtype = np.uint8)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30],outImg = img3,flags=2)

plt.imshow(img3),plt.show()



import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('C:/Users/situ.st.1/Pictures/box.png',0) # queryImage
img2 = cv2.imread('C:/Users/situ.st.1/Pictures/box_in_scene.png',0) # trainImage
# Initiate SIFT detector
sift = cv2.SIFT()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
# 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
# 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],flags=2)
plt.imshow(img3),plt.show()





















