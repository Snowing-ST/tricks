# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:28:23 2017

@author: situ.st.1
"""

import cv2
import copy
import numpy as np
import sys
import pandas as pd
import os
from matplotlib import pyplot as plt


def getMask(image,upper = (180,240,255),lower = (1,130,150),ero=19,dila = 17):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    plt.hist(gray.ravel(),256)  #明显看到呈双峰状态
#    plt.show()
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #自动选择阈值  
#    ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
#    plt.imshow(thresh),plt.show()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(ero,ero)) #因不同情况而异，可把形态学处理部分剥离
    erode = cv2.erode(thresh, element) #腐蚀
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(dila,dila))   
    dilate = cv2.dilate(thresh, element) #膨胀
#    plt.imshow(erode,"gray")
#    plt.show()
#    plt.imshow(dilate,"gray")
#    plt.show()
    #将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像  
    result = cv2.absdiff(dilate,erode)
    #上面得到的结果是灰度图，将其二值化以便更清楚的观察结果  
#    plt.imshow(result,"gray")
#    plt.show()
    img_fg = cv2.bitwise_and(image,image,mask = result) 
#    cv2.imshow('img_fg',img_fg)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    plt.imshow(img_fg,"gray")
#    plt.show()
#    upper = (180,240,255) #为1045改
#    lower = (1,130,150)
    hsv = cv2.cvtColor(img_fg, cv2.COLOR_BGR2HSV)
    hsv[(hsv[:,:,0]>9) & (hsv[:,:,0]<20)] = np.array([0,0,0],dtype = np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    result3 = cv2.merge([result,result,result])
    mask3 = cv2.merge([mask,mask,mask])
    org_dilate = cv2.addWeighted(image,1,result3,0.5,0)
    org_mask = cv2.addWeighted(image,1,mask3,0.5,0)
#    plt.imshow(mask,"gray")
#    plt.show()
    l,w,h = image.shape
    blank = np.zeros((l,w*3,h),np.uint8)
    blank[:,:w,:] = image
    blank[:,w:2*w,:] = org_dilate
    blank[:,2*w:3*w,:] = org_mask       
    return [mask,org_dilate,org_mask,blank]

def getrectangle2(num,X,Y,tooth = None,h=50,l=10):  # num is the No. of the gum, h is the height of the teeth
    if(num==0):#第一颗牙
        return([(X[0]-(X[1]-X[0])/2,Y[0]-h),(X[0]+(X[1]-X[0])/2,Y[0]+l)])
    elif(num<len(X)-1 and X[num]>X[num+1]):#第六个？
        return([(X[num]-(X[num]-X[num-1])/2,Y[num]-h),(X[num]+(X[num]-X[num-1])/2,Y[num]+l)])
    elif(X[num-1]>X[num]): #第2，3，4，5，7，8，9
        return([(X[num]-(X[num+1]-X[num])/2,Y[num]-l),(X[num]+(X[num+1]-X[num])/2,Y[num]+h)])
    elif(num==len(X)-1):#第10
        return([(X[num]-(X[num]-X[num-1])/2,Y[num]-l),(X[num]+(X[num]-X[num-1])/2,Y[num]+h)])
    elif ((num<5 and tooth==None) or (tooth!=None and tooth[num]=="upper_part")):#第1，2，3，4，5
        return([(X[num]-(X[num]-X[num-1])/2,Y[num]-h),(X[num]+(X[num+1]-X[num])/2,Y[num]+l)])
    else:
        return([(X[num]-(X[num]-X[num-1])/2,Y[num]-l),(X[num]+(X[num+1]-X[num])/2,Y[num]+h)])

def paintRect(img,fname,X,Y,drawtype,upper,lower,ero,dila,tooth=None,plot = False,check=False,h=50,l=10,w=1,minpix=15,minpix2=10):
    img1=copy.deepcopy(img)
    mask,org_dilate,org_mask,blank = getMask(img,upper,lower,ero,dila) 
    rect_df = pd.DataFrame()
    if np.sum(mask)/255>=minpix:
        if drawtype == "multiple":           
            for i in np.arange(len(X)):
                rec=getrectangle2(i,X,Y,tooth,h,l)
                p1=np.array(rec[0],dtype=np.int32)
                p2=np.array(rec[1],dtype=np.int32)
                temp_df = pd.DataFrame({"fname":[fname.split(".")[0]],
                                        "redlight1":[1],
                                        "tooth":[i+1],
                                        "redlight2":[0],
                                        "pixel":[0]},
                                        columns = ["fname","redlight1","tooth","redlight2","pixel"])                
                if np.sum(mask[p1[1]:p2[1],p1[0]:p2[0]])/255>minpix2:
    #                cv2.rectangle(mask,tuple(p1),tuple(p2),(255,0,0),w)
                    cv2.rectangle(org_mask,tuple(p1),tuple(p2),(255,0,0),w)
                    temp_df["redlight2"] = 1
                    temp_df["pixel"] = np.sum(mask[p1[1]:p2[1],p1[0]:p2[0]])/255
                rect_df = pd.concat([rect_df,temp_df])
        if drawtype == "max":
            redlight=[]
            rect_df = pd.DataFrame({"fname":[fname.split(".")[0]],
                                    "redlight1":[0],
                                    "tooth":[0],
                                    "redlight2":[0],
                                    "pixel":[0]},
                                    columns = ["fname","redlight1","tooth","redlight2","pixel"])                        
            for i in np.arange(len(X)): 
                rec=getrectangle2(i,X,Y,tooth,h,l)
                p1=np.array(rec[0],dtype=np.int32)
                p2=np.array(rec[1],dtype=np.int32)
                redlight.append(np.sum(mask[p1[1]:p2[1],p1[0]:p2[0]])/255)
                #print redlight
            if max(redlight)>0:
                In=redlight.index(max(redlight))
                rec=getrectangle2(In,X,Y,tooth,h,l)
                p1=np.array(rec[0],dtype=np.int32)
                p2=np.array(rec[1],dtype=np.int32)
                cv2.rectangle(org_mask,tuple(p1),tuple(p2),(0,0,255),w)
                rect_df["redlight1"] = 1
                rect_df["redlight2"] = 1
                rect_df["tooth"] = In+1
                rect_df["pixel"] = max(redlight)
    if np.sum(mask)/255<minpix:
#        print "picture "+fname+" has no redlight"
        rect_df = pd.DataFrame({"fname":[fname.split(".")[0]],
                                                "redlight1":[0],
                                                "tooth":[0],
                                                "redlight2":[0],
                                                "pixel":[np.sum(mask)/255]},
                                                columns = ["fname","redlight1","tooth","redlight2","pixel"])                        
    if plot == True:        
        if check ==True:
            blank[:,:img1.shape[1],:] = img1
            return [rect_df,blank]
        if check == False:
            return [rect_df,org_mask]
    if plot == False:
        return rect_df
    
def kp(MskDir,kpname):      
        X=[]
        Y=[]
        tooth = []
        i=0
        f=open(MskDir+kpname)
        line = f.readline()
        while(1):
            i=i+1
            #print (len(X))
            if(len(line)>3):
            #print (len(line))
                spl=line.split('\t')
                X.append(float(spl[0]))
                Y.append(float(spl[1]))
                if len(spl)==3:
                    tooth.append(spl[2].split("\n")[0])            
            line = f.readline()
            if(i>15):
                break
        f.close()
        if tooth==[]:
            tooth = None
        return X,Y,tooth
    
    
myfolder =  "C:/Users/situ.st.1/Desktop/SituXueying/8.3/"
#myfolder = "/raid/home/xujun/tooth_keypoint/1016a002/"
fname = "cropped.jpg"
kpname = "kp.txt"
toothImg = cv2.imread(myfolder+fname)
X,Y,tooth = kp(myfolder,kpname)
print X,Y,tooth


def def_func(x):
    pass  #不做任何事情，一般用做占位语句
    
img = np.zeros((512,384,3), np.uint8)  
cv2.namedWindow('image')  
  
cv2.createTrackbar("H_min",'image',0,180,def_func)
cv2.createTrackbar("S_min",'image',0,255,def_func)
cv2.createTrackbar("V_min",'image',0,255,def_func)
cv2.createTrackbar("minpix","image",0,20,def_func)
cv2.createTrackbar("minpix2","image",0,20,def_func)
cv2.createTrackbar("erode","image",0,30,def_func)
cv2.createTrackbar("dilate","image",0,30,def_func)

  
switch = '0:OFF\n1:ON'  
cv2.createTrackbar(switch,'image',0,1,def_func)  
  
while(True):  
    cv2.imshow('image', img)  
    k = cv2.waitKey(1)&0xff  
    if k == 27:  
        break  
    H_min = cv2.getTrackbarPos('H_min', 'image')  
    S_min = cv2.getTrackbarPos('S_min', 'image')  
    V_min = cv2.getTrackbarPos('V_min', 'image')
    minpix = cv2.getTrackbarPos("minpix","image")
    minpix2 = cv2.getTrackbarPos("minpix2","image")
    ero = cv2.getTrackbarPos("erode","image")
    dila = cv2.getTrackbarPos("dilate","image")
    
    s = cv2.getTrackbarPos(switch, 'image')  
  
    if s == 0:  
        img[:] = toothImg  
    else:  
        lower = (H_min,S_min,V_min)
        upper = (180,255,255)
        img = paintRect(toothImg,fname,X,Y,drawtype="multiple",upper=upper,lower=lower,ero=ero,dila=dila,tooth=tooth,plot = True,check=False,minpix=minpix,minpix2=minpix2)[1]

cv2.destroyAllWindows() 