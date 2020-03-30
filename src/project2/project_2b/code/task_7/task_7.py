import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 


import numpy as np
import cv2
import glob
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROWS = 9
COLS = 6
N = 11

#Change the following lines to match your system
base_folder = "/home/laukik/Perception-projects/src/project2/project_2b/"
param_folder="/home/laukik/Perception-projects/src/project2/project_2a/parameters/"


left_path="images/task_7/left_"
right_path="images/task_7/right_"



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

#Load the camera params now (Only the left camera is loaded:
left_int = pd.read_csv(param_folder+'left_camera_intrinsics.csv')
mtxl = left_int['Camera Matrix'][0]
mtxl = df_to_param(mtxl, mat=1)
distl = left_int['Distortion Coefficient'][0]
distl = df_to_param(distl, mat=1)


b=62 #In mm units
fsx=423.27384816 #Obtained from the camera matrix



for i in range(10):
    left=base_folder+left_path+str(i)+'.png'
    right=base_folder+left_path+str(i+1)+'.png' #The 'right' image is taken by the left camera at a later instant

    imgL = cv2.imread(left)
    imgL=cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgL=cv2.undistort(imgL, mtxl,distl, None,None)   #This is the undistorted image

    imgR = cv2.imread(right)
    imgR=cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgR=cv2.undistort(imgR, mtxl,distl, None,None)   #This is the undistorted image

    orb_L = cv2.ORB_create()
    keypoints_L, descriptor_L = orb_L.detectAndCompute(imgL, None)

    orb_R = cv2.ORB_create()
    keypoints_R, descriptor_R = orb_R.detectAndCompute(imgR, None)

    Keypoints_Image_L   =   cv2.drawKeypoints(  imgL,keypoints_L,None,color=[0,255,0],flags=cv2.DrawMatchesFlags_DEFAULT)
    #plt.imshow(Keypoints_Image_L)#,plt.show()
    
    Keypoints_Image_R   =   cv2.drawKeypoints(  imgR,keypoints_R,None,color=[0,0,150],flags=cv2.DrawMatchesFlags_DEFAULT)
    #plt.imshow(Keypoints_Image_R)#,plt.show()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptor_L,descriptor_R)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(imgL,keypoints_L,imgR,keypoints_R,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    #This is matched without outlier masking
    #plt.imshow(img3)
    #plt.title('Without Inlier Calculation')
    #plt.show()
    

    left_pts = np.float32([keypoints_L[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    right_pts = np.float32([keypoints_R[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    essential_matrix=cv2.findEssentialMat(left_pts,right_pts,mtxl)

    mask=essential_matrix[1]
    essential_matrix=essential_matrix[0] #Only the matrix is needed

    '''
    inlier_matches=[]
    for i in range(len(matches)):
        if(mask[i][0]==1):
            inlier_matches.append(matches[i])
    '''

    #This considers only inlier matches
    inlier_img3=cv2.drawMatches(imgL,keypoints_L,imgR,keypoints_R,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchesMask=mask.ravel().tolist())
    
    #plt.imshow(inlier_img3)
    #plt.title('Inlier matching')
    #plt.show()

    info=cv2.recoverPose(essential_matrix,   left_pts,right_pts)

    #The rotation and translation are obtained as follows
    R=info[1]
    t=info[2]

    #Projection matrices
    P1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    P2=R.tolist()
    P2[0].append(t[0][0])
    P2[1].append(t[1][0])
    P2[2].append(t[2][0])
    P2=np.array(P2)

    pts4D=cv2.triangulatePoints(P1,P2,left_pts,right_pts)
    pts3D = pts4D[:3,:]/np.repeat(pts4D[3,:], 3).reshape(3,-1) #Obtain the points in 3D

    Ys = pts3D[1,:]
    Zs = pts3D[2,:]
    Xs = pts3D[0,:]


    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    '''
    
    '''
    #Display camera poses
    C1_start=[0,0,0]
    C1_end=[0,0.06,0]

    C2_start=t
    C2_end=t+np.matmul(R,C1_end)

    cam_x_start=[C1_start[0],C2_start[0][0]]
    cam_y_start=[C1_start[1],C2_start[1][0]]
    cam_z_start=[C1_start[2],C2_start[2][0]]

    cam_x_end=[C1_end[0],C2_end[0][0]]
    cam_y_end=[C1_end[1],C2_end[1][0]]
    cam_z_end=[C1_end[2],C2_end[2][0]]
    '''

    '''
    ax.scatter(Xs, Ys, Zs, c='r', marker='o')
    #ax.scatter(cam_x_start,cam_y_start,cam_z_start,c='b',marker='o')
    #ax.scatter(cam_x_end,cam_y_end,cam_z_end,c='g',marker='o')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    plt.title('3D point cloud: Use pan axes button below to inspect')
    #plt.show()
    '''

    import ipdb
    ipdb.set_trace()
    
    #plt.show()




