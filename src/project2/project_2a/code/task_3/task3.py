# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:42:12 2020

@author: ishani
"""
"""Import necessary packages"""

import numpy as np
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import glob
from matplotlib import pyplot as plt
import copy

from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from skimage import data, img_as_float
import ipdb

import pandas as pd

ROWS = 9
COLS = 6
N = 11

"""Step (1): Load the images and camera parameters. """

path="/home/laukik/Perception-projects/src/project2/project_2a"
left = glob.glob(path+"/images/task_1/left_*.png")
left.sort()
right = glob.glob(path+"/images/task_1/right_*.png")
right.sort()


#left = glob.glob(r"C:\Users\ishan\Downloads\project_2a\project_2a\images\task_1\left_*.png")
#left.sort()
#right = glob.glob(r"C:\Users\ishan\Downloads\project_2a\project_2a\images\task_1\right_*.png")
#right.sort()


obj_p = np.zeros((ROWS*COLS, 3), np.float32)
obj_p[:,:2] = np.mgrid[0:ROWS, 0:COLS].T.reshape(-1, 2)

obj_points_left, img_points_left = [], []
obj_points_right, img_points_right = [], []

for i in range(0, N):
    img1 = cv2.imread(left[i])
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret_left, corner_left = cv2.findChessboardCorners(img2, (ROWS, COLS), None)

    img3 = cv2.imread(right[i])
    img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY) #gray 
    ret_right, corner_right = cv2.findChessboardCorners(img4, (ROWS, COLS), None)

    obj_points_left.append(obj_p)
    img_points_left.append(corner_left)
    obj_points_right.append(obj_p)
    img_points_right.append(corner_right)

retl, mtxl, distl, rvecsl, tvecsl = cv2.calibrateCamera(obj_points_left, img_points_left, img2.shape[::-1], None, None)
retr, mtxr, distr, rvecsr, tvecsr = cv2.calibrateCamera(obj_points_right, img_points_right, img4.shape[::-1], None, None)
    


"""Step (2): Detect features."""

print(img2.shape, img4.shape)
mapXL, mapYL = cv2.initUndistortRectifyMap(mtxl, distl, None, None, img2.shape, 5)
mapXR, mapYR = cv2.initUndistortRectifyMap(mtxr, distr, None, None, img4.shape, 5)

undistorted_left=[]
undistorted_right=[]

"""detect ORB feature points on each image, using the OpenCV library "ORB" class."""

#left = glob.glob(r"C:\Users\ishan\Downloads\project_2a\project_2a\images\task_3_and_4\left_*.png")
#left.sort()
#right = glob.glob(r"C:\Users\ishan\Downloads\project_2a\project_2a\images\task_3_and_4\right_*.png")
#right.sort()

path="/home/laukik/Perception-projects/src/project2/project_2a/images/task_3_and_4/"
left = glob.glob(path+"/left_*.png")
left.sort()
right = glob.glob(path+"/right_*.png")
right.sort()

for i in range(0,N):
    #print(i)
    imgL = cv2.imread(left[i])
    
    imgL=cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgL=cv2.undistort(imgL, mtxl,distl, None,None)   #This is the undistorted image
    orb = cv2.ORB_create()
    keypoints_L, descriptor_L = orb.detectAndCompute(imgL, None)
    imgR = cv2.imread(right[i])
    imgR=cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgR=cv2.undistort(imgR, mtxr,distr, None,None)   #This is the undistorted image
    orb_R = cv2.ORB_create()
    keypoints_R, descriptor_R = orb.detectAndCompute(imgR, None)
    
    Keypoints_Image_L	=	cv2.drawKeypoints(	imgL,keypoints_L,None,color=[0,255,0],flags=cv2.DrawMatchesFlags_DEFAULT)
    plt.imshow(Keypoints_Image_L)#,plt.show()
    
    Keypoints_Image_R	=	cv2.drawKeypoints(	imgR,keypoints_R,None,color=[0,0,150],flags=cv2.DrawMatchesFlags_DEFAULT)
    plt.imshow(Keypoints_Image_R)#,plt.show()
    coordinates_L = peak_local_max(imgL, min_distance=20)
    coordinates_R = peak_local_max(imgR, min_distance=20)
#    Keypoints_Image_L	=	cv2.drawKeypoints(	imgL,coordinates_L,None,color=[0,255,0],flags=cv2.DrawMatchesFlags_DEFAULT)
#    plt.imshow(Keypoints_Image_L),plt.show()
#    
#    Keypoints_Image_R	=	cv2.drawKeypoints(	imgR,coordinates_R,None,color=[0,0,150],flags=cv2.DrawMatchesFlags_DEFAULT)
#    plt.imshow(Keypoints_Image_R),plt.show()
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptor_L,descriptor_R)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(imgL,keypoints_L,imgR,keypoints_R,matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)#,plt.show()

    
#Part 5 (Laukik has written the following lines)

def df_to_param(x, mat = 0):
    n = len(x)
    check = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', 'e']
    x = x[1:n-1]
    out = []
    i = 0
    while(i<n-2):
        if(x[i] == '['):
            new = []
        if(x[i] in check):
            temp = x[i]
            i += 1
            while(x[i] in check):
                temp += x[i]
                i += 1
            temp = float(temp)
            new.append(temp)
        elif(x[i] == ']'):
            out.append(new)
            i+=1
        else:
            i+=1
    print(out)
    if(len(out) == 1):
        out = out[0]
    if(mat == 1):
        return np.matrix(out)
    elif(mat == 0):
        return out




path="/home/laukik/Perception-projects/src/project2/project_2a/" #@ishani you should change this

stereo_rect=pd.read_csv(path+'parameters/stereo_rectification.csv')
R1=stereo_rect['Rotation 1'][0]
R1=df_to_param(R1,mat=1)

R2=stereo_rect['Rotation 2'][0]
R2=df_to_param(R2,mat=1)

p1=stereo_rect['Pose 1'][0]
p1=df_to_param(p1,mat=1)

p2=stereo_rect['Pose 2'][0]
p2=df_to_param(p2,mat=1)

ipdb.set_trace()



    
    
   