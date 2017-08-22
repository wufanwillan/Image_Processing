#!/usr/bin/env python
import cv2
import numpy as np
import os
import config
import copy
from abc import ABCMeta, abstractmethod
import skimage

class ImageMask(object):

    __metaclass__=ABCMeta

    def __init__(self,image):
        self.imagelist=copy.deepcopy(image)
        self.maskedimg=np.zeros_like(image)
        self.gaussiankernelsize=config.GaussianKernelSize
        self.meankernelsize=config.MeanKernelSize
        self.lowthreshold=config.LowThreshold
        self.highthreshold=config.HighThreshold
        self.greenmaskup=config.GreenMaskUp
        self.enhencelow=config.EnhenceLow
        self.enhencehigh=config.EnhenceHigh

    def Combine_Mask(self):
        imagenormal=map(self.Image_Normalize,self.imagelist)
        imagemasked=map(self.Convert_Image,imagenormal)
        imageeaged=map(self.Apply_Canny,imagenormal)
        imagecombined=map(cv2.bitwise_and,imagemasked,imageeaged)
        try:
            imagewithmask=reduce(cv2.bitwise_or,imagecombined)
        except:
            imagewithmask=imagecombined
        #imagecombined=map(self._Gaussian_Blur,imagecombined)
        #imagecombined=map(self._Conv_Kernel,imagecombined)
        imagecombined=map(self._ImageEnhence,imagecombined)
        imagecombined=map(self._select_region,imagecombined)
        return imagecombined

       # return imagemasked
    def _Gray_Scaling(self,image):
        return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)


    def _Gaussian_Blur(self,image,gaussiankernelsize=config.GaussianKernelSize):
        return cv2.GaussianBlur(image,(gaussiankernelsize,gaussiankernelsize),0)

    def _Canny(self,image,low=config.LowThreshold,high=config.HighThreshold):
        return cv2.Canny(image,low,high)

    def _ImageEnhence(self,image,low=config.EnhenceLow,high=config.EnhenceHigh):
        return cv2.inRange(image,low,high)

    def Apply_Canny(self,image):
        image=self._Gray_Scaling(image)
        image=self._Canny(image)
        image=self._Gaussian_Blur(image)
        image=self._ImageEnhence(image)
        return image

    def _Conv_Kernel(self,image,meankernelsize=config.MeanKernelSize):
        return cv2.blur(image,(meankernelsize,meankernelsize))

    def Image_Normalize(self,image):
        imagegray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        graymax1=map(max,imagegray)
        graymax2=max(graymax1)
        image=self.Convert_Type(image)
        image=graymax2*image/255.0
        image=self.Convert_Type(image)
        return image

    def _filter_region(self,image, vertices):
    	mask = np.zeros_like(image)
        if len(mask.shape)==2:
            cv2.fillPoly(mask, vertices, 255)
        else:
            cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
        return cv2.bitwise_and(image, mask)

    def _select_region(self,image):
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.0, rows*1.0]
        top_left     = [cols*0.0, rows*0.6]
        bottom_right = [cols*1.0, rows*1.0]
        top_right    = [cols*1.0, rows*0.6]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        return self._filter_region(image, vertices)

    def Hough_Line(self,image):
        return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

    def Convert_Type(self,image):
        if image.dtype=='float64':
            image[image>1.0]=1.0
            image[image<0.0]=0.0
            image=skimage.img_as_ubyte(image)
        elif image.dtype=='uint8':
            image=skimage.img_as_float(image)
        else:
            print 'Type of image not provide'
        return image


    @abstractmethod
    def Convert_Image(self,emage):
        pass

class RGBWhiteImageMask(ImageMask):
    def __init__(self,image):
        ImageMask.__init__(self,image)
        self.whitemasklow=config.WhiteMaskLow
        self.whitemaskup=config.WhiteMaskUp

    def Convert_Image(self,image):
        whitemaskedimg=cv2.inRange(image,self.whitemasklow,self.whitemaskup)
        return whitemaskedimg

class RGBGreenImageMask(ImageMask):
    def __init__(self,image):
        ImageMask.__init__(self,image)
        self.GreenMaskLow=config.GreenMaskLow
        self.GreenMaskUp=config.GreenMaskUp

    def Convert_Image(self,image):
        Green_Maskedimg=cv2.inRange(image,config.GreenMaskLow,config.GreenMaskUp)
        return Green_Maskedimg

class HSVImageMask(ImageMask):
    def __init__(self,image):
        ImageMask.__init__(self,image)
        self.HSVMaskLow=config.HSVMaskLow
        self.HSVMaskUp=config.HSVMaskUp

    def Convert_Image(self,image):
        image=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        HSV_Maskedimg=cv2.inRange(image,self.HSVMaskLow,self.HSVMaskUp)
        return HSV_Maskedimg

class HLSImageMask(ImageMask):
    def __init__(self,image):
        ImageMask.__init__(self,image)
        self.HLSMaskLow=config.HLSMaskLow
        self.HLSMaskUp=config.HLSMaskUp

    def Convert_Image(self,image):
        image=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        HLS_Maskedimg=cv2.inRange(image,self.HLSMaskLow,self.HLSMaskUp)
        return HLS_Maskedimg



