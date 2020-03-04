import numpy as np

#Import cv2 without conflicting with ROS
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import glob
import os

from matplotlib import pyplot as plt

import ipdb

ROWS = 9
COLS = 6
N = 11


path="/home/laukik/Perception-projects/src/project2/project_2a/images/task_1"
left = glob.glob(path+"/left_*.png")
left.sort()
right = glob.glob(path+"/right_*.png")
right.sort()


obj_p = np.zeros((ROWS*COLS, 3), np.float32)
obj_p[:,:2] = np.mgrid[0:ROWS, 0:COLS].T.reshape(-1, 2)

obj_points_left, img_points_left = [], []
obj_points_right, img_points_right = [], []

for i in range(0, N):
    #ipdb.set_trace()
    img1 = cv2.imread(left[i])
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret_left, corner_left = cv2.findChessboardCorners(img2, (ROWS, COLS), None)

    img3 = cv2.imread(right[i])
    img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    ret_right, corner_right = cv2.findChessboardCorners(img4, (ROWS, COLS), None)

    obj_points_left.append(obj_p)
    img_points_left.append(corner_left)
    obj_points_right.append(obj_p)
    img_points_right.append(corner_right)


retl, mtxl, distl, rvecsl, tvecsl = cv2.calibrateCamera(obj_points_left, img_points_left, img2.shape[::-1], None, None)
retr, mtxr, distr, rvecsr, tvecsr = cv2.calibrateCamera(obj_points_right, img_points_right, img4.shape[::-1], None, None)
    
#Step 4
print(img2.shape, img4.shape)
mapXL, mapYL = cv2.initUndistortRectifyMap(mtxl, distl, None, None, img2.shape, 5)
mapXR, mapYR = cv2.initUndistortRectifyMap(mtxr, distr, None, None, img4.shape, 5)

undistorted_left=[]
undistorted_right=[]
#Start block matching
path="/home/laukik/Perception-projects/src/project2/project_2a/images/task_3_and_4"
left = glob.glob(path+"/left_*.png")
left.sort()
right = glob.glob(path+"/right_*.png")
right.sort()

matcher=cv2.StereoBM_create(numDisparities=48,blockSize=15) #Arbitrary choice
matcher.setPreFilterType(1)
disparities=[]
for i in range(0,N):
    #print(i)
    imgL = cv2.imread(left[i])
    
    imgL=cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgL=cv2.undistort(imgL, mtxl,distl, None,None)   #This is the undistorted image

    imgR = cv2.imread(right[i])
    imgR=cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgR=cv2.undistort(imgR, mtxr,distr, None,None)   #This is the undistorted image    
    #The following are undistorted images
    disparities.append(matcher.compute(imgL,imgR))
    

#Choose to display scenes 4 and 6
b=62 #In mm units
fsx=mtxl[0][0]
depth_img_4=np.true_divide(b*fsx,disparities[4])
plt.figure('image_4')
plt.imshow(depth_img_4,'gray')
plt.show()

plt.figure('disparity image4')
plt.imshow(disparities[4],'gray')
plt.show()



depth_img_6=np.true_divide(b*fsx,disparities[6])
plt.figure('image_6')
plt.imshow(depth_img_6,'gray')
plt.show()

plt.figure('disparity image6')
plt.imshow(disparities[6],'gray')
plt.show()


