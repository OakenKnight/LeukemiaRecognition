#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as bb8
import cv2
from scipy import signal


# In[2]:


# iscrtavanje slika i plotova unutar samog browsera
#%matplotlib inline 
# prikaz vecih slika 
#matplotlib.rcParams['figure.figsize'] = 16,12


# In[3]:


def brightness_up(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - 30
    v[v > lim] = 255
    v[v <= lim] += 30

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


# In[4]:


def pixelVal(pix, r1, s1, r2, s2): 
    if (0 <= pix and pix <= r1): 
        return (s1 / r1) * pix 
    elif (r1 < pix and pix <= r2): 
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1 
    else: 
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2 


# In[5]:


image = cv2.imread("dataset/train/img-9.jpg") 
image = brightness_up(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

blurM = cv2.medianBlur(gray, 5) 
plt.imshow(blurM)  


# In[ ]:





# In[6]:


r1 = 120
s1 = 0
r2 = 200
s2 = 255


# In[7]:


pixelVal_vec = bb8.vectorize(pixelVal) 
  
# Apply contrast stretching.  
contrast_stretched = pixelVal_vec(gray, r1, s1, r2, s2) 
contrast_stretched_blurM = pixelVal_vec(blurM, r1, s1, r2, s2) 
plt.imshow(contrast_stretched,'gray')


# In[8]:


plt.imshow(contrast_stretched_blurM,'gray')


# In[9]:


ret, image_bin = cv2.threshold(contrast_stretched, 170, 255, cv2.THRESH_BINARY) # ret je vrednost praga, image_bin je binarna slika
plt.imshow(image_bin,'gray')


# In[10]:


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
dilation = cv2.dilate(image_bin,kernel,iterations = 2)

dilation = cv2.erode(dilation,kernel,iterations = 4)
plt.imshow(dilation,'gray')
plt.imshow(image)


# In[11]:


plt.imshow(dilation,'gray')

