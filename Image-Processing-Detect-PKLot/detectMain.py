#!/usr/bin/env python
import cv2
import numpy as np
import os
import config
from detectPKLot2 import ImageMask,RGBWhiteImageMask,RGBGreenImageMask,HLSImageMask,HSVImageMask

def get_Image():
	image=[]
	for seq in range(len(config.FilePath)):
	    image.append(cv2.imread(config.FilePath[seq]))
	return image


image=get_Image()
#red,green,blue=image[0]
#print red.shape
#image=image[0]
#print max(image.reshape(-1,1))
#image1=cv2.cvtColor(image[0],cv2.COLOR_RGB2HLS)
#image2=cv2.cvtColor(image[0],cv2.COLOR_RGB2HSV)


ImageMaskedobj=RGBWhiteImageMask(image)
ImageMasked=ImageMaskedobj.Combine_Mask()
#img1=map(ImageMaskedobj.Convert_Type,image)
img2=map(ImageMaskedobj.Convert_Type,ImageMasked)
#img3=img1[0]
#img3[img2[0]>0]=1
#ImageMaskedobj=RGBWhiteImageMask(image)
img2=map(ImageMaskedobj.Convert_Type,img2)
img3=map(ImageMaskedobj.Hough_Line,img2)
#ImageMaskedobj=RGBWhiteImageMask(image)
#ImageMasked=ImageMaskedobj.Combine_Mask()
#print type(img3)
#print len(img3)
print img3[0].shape
#img1=map(ImageMasked.Convert_Type,img1)
#ImageMasked=ImageMasked.Image_Normalize(image[0])
#Interference of HSV and HLS transform
#ImageMasked=HSVImageMask(image)
#image1=ImageMasked.Convert_Image(image[0])
#ImageMasked=HLSImageMask(image)
#cv2.imshow('white1',ImageMasked[0][1])
#cv2.imshow('white2',ImageMasked[0][2])
for line in img3[0]:
    for x1,x2,y1,y2 in line:
        print x1,x2,y1,y2

#cv2.imshow('img0',ImageMasked[0])
#cv2.imshow('img1',image[0])
#cv2.imshow('img1',image1[:,:,0])
#cv2.imshow('img2',image1[:,:,1])
#cv2.imshow('img3',img3[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

