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
import pandas as pd
import ipdb

ROWS = 9
COLS = 6
N = 11

'''
path="/home/laukik/Perception-projects/src/project2/project_2a/images/task_1"
left = glob.glob(path+"/left_*.png")
n_images=len(left)
left,right=[],[]
for i in range(n_images):
    left.append(path+'/left_'+str(i)+'.png')
    right.append(path+'/right_'+str(i)+'.png')


obj_p = np.zeros((ROWS*COLS, 3), np.float32)
obj_p[:,:2] = np.mgrid[0:ROWS, 0:COLS].T.reshape(-1, 2)

obj_points_left, img_points_left = [], []
obj_points_right, img_points_right = [], []

for i in range(0, N):
    
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
'''


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

path="/home/laukik/Perception-projects/src/project2/project_2a/"

base_folder = path
    
left_int = pd.read_csv(base_folder+'parameters/left_camera_intrinsics.csv')
mtxl = left_int['Camera Matrix'][0]
mtxl = df_to_param(mtxl, mat=1)
distl = left_int['Distortion Coefficient'][0]
distl = df_to_param(distl, mat=1)


right_int = pd.read_csv(base_folder+'parameters/right_camera_intrinsics.csv')
mtxr = right_int['Camera Matrix'][0]
mtxr = df_to_param(mtxr, mat=1)
distr = right_int['Distortion Coefficient'][0]
distr = df_to_param(distr, mat=1)

#retl, mtxl, distl, rvecsl, tvecsl = cv2.calibrateCamera(obj_points_left, img_points_left, img2.shape[::-1], None, None)
#retr, mtxr, distr, rvecsr, tvecsr = cv2.calibrateCamera(obj_points_right, img_points_right, img4.shape[::-1], None, None)


#Step 4
#print(img2.shape, img4.shape)
#mapXL, mapYL = cv2.initUndistortRectifyMap(mtxl, distl, None, None, img2.shape, 5)
#mapXR, mapYR = cv2.initUndistortRectifyMap(mtxr, distr, None, None, img4.shape, 5)

undistorted_left=[]
undistorted_right=[]

#Start block matching
path+="/images/task_3_and_4"
left = glob.glob(path+"/left_*.png")
n_images=len(left)
left,right=[],[]
for i in range(n_images):
    left.append(path+'/left_'+str(i)+'.png')
    right.append(path+'/right_'+str(i)+'.png')

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
    #ipdb.set_trace()   

#Choose to display scenes 4 and 6
b=62 #In mm units
fsx=423.27384816 #Obtained from the camera matrix
depth_img_7=np.true_divide(b*fsx,disparities[7])
plt.figure('depth_7')
plt.imshow(depth_img_7,'gray')
plt.show()

plt.figure('disparity image7')
plt.imshow(disparities[7],'gray')
plt.show()



depth_img_6=np.true_divide(b*fsx,disparities[4])
plt.figure('depth_4')
plt.imshow(depth_img_6,'gray')
plt.show()

plt.figure('disparity image4')
plt.imshow(disparities[4],'gray')
plt.show()


