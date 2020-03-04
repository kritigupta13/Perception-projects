import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd

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

ROWS = 9
COLS = 6
base_folder = "D:/ASU/CSE598 - Perception in Robotics/project_2a/"

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

#Step 1 Init
print("-- Task 2 --")
left2 = glob.glob(base_folder+"images/task_2/left_*.png")
right2 = glob.glob(base_folder+"images/task_2/right_*.png")

obj_p2 = np.zeros((ROWS*COLS, 3), np.float32)
obj_p2[:,:2] = np.mgrid[0:ROWS, 0:COLS].T.reshape(-1, 2)
obj_p2 = np.array([obj_p2])
obj_points_left2, img_points_left2 = [], []
obj_points_right2, img_points_right2 = [], []

#Step 1 + 2
img1 = cv2.imread(left2[0])
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret_left21, corner_left21 = cv2.findChessboardCorners(img2, (ROWS, COLS), None)
img3 = cv2.imread(right2[0])
img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
ret_right21, corner_right21 = cv2.findChessboardCorners(img4, (ROWS, COLS), None)
temp11 = np.array([[corner for [corner] in corner_left21]])
temp21 = np.array([[corner for [corner] in corner_right21]])

img1 = cv2.imread(left2[1])
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret_left2, corner_left22 = cv2.findChessboardCorners(img2, (ROWS, COLS), None)
img3 = cv2.imread(right2[1])
img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
ret_right2, corner_right22 = cv2.findChessboardCorners(img4, (ROWS, COLS), None)
temp12 = np.array([[corner for [corner] in corner_left22]])
temp22 = np.array([[corner for [corner] in corner_right22]])

#Step 3
term_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
retval, camMat1, distC1, camMat2, distC2, R, T, E, F = cv2.stereoCalibrate(obj_p2, temp11, temp12, 
                           mtxl, distl, mtxr, distr, img2.shape[::-1], flags = cv2.CALIB_FIX_INTRINSIC,
                           criteria=term_criteria)

stereo_cal = {'Camera Matrix 1': [camMat1], 
              'Distortion Coefficient 1': [distC1], 
              'Camera Matrix 2': [camMat2], 
              'Distortion Coefficient 2': [distC2], 
              'R': [R], 'T': [T], 'E': [E], 'F': [F]}
df_cal = pd.DataFrame(data = stereo_cal)
df_cal.to_csv(base_folder+'parameters/stereo_calibration.csv', index = False)

result_left0 = cv2.undistortPoints(temp11, mtxl, distl)
result_left1 = cv2.undistortPoints(temp12, mtxr, distr)

result_left0 = result_left0.reshape(2,54)
result_left1 = result_left1.reshape(2,54)

P1 = np.identity(4)
P2 = np.hstack((R, T))

points = cv2.triangulatePoints(P1[0:3], P2, result_left0, result_left1)

cpoints = np.true_divide(points[:3,:],points[-1,:])
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(cpoints[0,:], cpoints[1,:], cpoints[2,:], c=cpoints[2,:], cmap='viridis')
plt.draw()
'''
rotation1, rotation2, pose1, pose2 = cv2.stereoRectify(camMat1, distC1, camMat2, distC2, img2.shape[::-1], R, T)[0:4]

stereo_rect = {'Rotation 1': [rotation1], 'Rotation 2': [rotation2], 'Pose 1': [pose1], 'Pose 2': [pose2]}
df_rect = pd.DataFrame(data = stereo_rect)
df_rect.to_csv(base_folder+'parameters/stereo_rectification.csv', index = False)

mapX0, mapY0 = cv2.initUndistortRectifyMap(camMat1, distC1, None, None, img2.shape, 5)
mapX1, mapY1 = cv2.initUndistortRectifyMap(camMat2, distC2, None, None, img2.shape, 5)

img0 = cv2.imread(left2[0])
img1 = cv2.imread(left2[1])
dst0 = cv2.remap(img0, mapX0, mapY0, cv2.INTER_LINEAR)
dst1 = cv2.remap(img1, mapX1, mapY1, cv2.INTER_LINEAR)

cv2.imwrite(base_folder+'output/task_2/leftRect0.png', dst0)
cv2.imwrite(base_folder+'output/task_2/leftRect1.png', dst1)
print("DONE TASK 2")
