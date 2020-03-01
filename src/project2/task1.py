import numpy as np
import cv2
import glob

ROWS = 9
COLS = 6
N = 11

left = glob.glob("/home/kriti/catkin_ws/src/project_2a/images/task_1/left_*.png")
right = glob.glob("/home/kriti/catkin_ws/src/project_2a/images/task_1/right_*.png")

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

retl, mtxl, distl, rvecsl, tvecsl = cv2.calibrateCamera(obj_points_left, img_points_left, img2.shape[::-1], None, None)
retr, mtxr, distr, rvecsr, tvecsr = cv2.calibrateCamera(obj_points_right, img_points_right, img4.shape[::-1], None, None)

#Step 4
print(img2.shape, img4.shape)
mapXL, mapYL = cv2.initUndistortRectifyMap(mtxl, distl, None, None, img2.shape, 5)
mapXR, mapYR = cv2.initUndistortRectifyMap(mtxr, distr, None, None, img4.shape, 5)

for i in range(0,N):
    print(i)
    imgL = cv2.imread(left[i])
    imgR = cv2.imread(right[i])
    dst_left = cv2.remap(imgL, mapXL, mapYL, cv2.INTER_LINEAR)
    dst_right = cv2.remap(imgR, mapXR, mapYR, cv2.INTER_LINEAR)

    cv2.imwrite('leftMAP_'+str(i)+'.png', dst_left)
    cv2.imwrite('rightMAP_'+str(i)+'.png', dst_right)
print("DONE")
