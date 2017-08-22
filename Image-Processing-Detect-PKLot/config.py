#!/usr/bin/env python

import os
import glob
import cv2
import numpy as np

##Defenition

##

##Execute

CurrentPath=os.getcwd()
ImagePath=os.path.join(CurrentPath,'Image')
FilePath=glob.glob(ImagePath+'/*.png')

GaussianKernelSize=9
MeanKernelSize=5
LowThreshold=40
HighThreshold=330
EnhenceLow=50
EnhenceHigh=255


WhiteMaskLow=np.uint8([50,50,30])
WhiteMaskUp=np.uint8([230,170,170])

GreenMaskLow=np.uint8([0,0,0])
GreenMaskUp=np.uint8([80,80,65])

HSVMaskLow=np.uint8([0,0,0])
HSVMaskUp=np.uint8([70,255,255])

HLSMaskLow=np.uint8([120,20,70])
HLSMaskUp=np.uint8([150,40,9])
