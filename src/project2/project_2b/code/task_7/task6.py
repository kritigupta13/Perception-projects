import numpy as np
import cv2
from cv2 import aruco
import glob
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

relative_path = '../../images/task_6/'

def plot_pyramid(axis,R,T):
    v = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 0]])
    v = v.T
    v = np.append(v, [[1, 1, 1, 1, 1]], axis=0)
    H = np.hstack((R, T))
    H = np.append(H, [[0, 0, 0, 1]], axis=0)
    v_t = (np.matrix(H) * np.matrix(v))
    v_t = np.delete(v_t, 3, 0)
    v = np.array(v_t.T)

    vertices = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
             [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]
    axis.add_collection3d(Poly3DCollection(vertices,facecolors='white', linewidths=1, edgecolors='k', alpha=.25))

def plot_square(axis):
    s_pts = np.array([[0, 5, 0], [5, 5, 0], [5, 0, 0], [0, 0, 0]])
    sqr_verts = [[s_pts[0], s_pts[1], s_pts[2], s_pts[3]]]
    m_pts = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]])
    m_verts = [[m_pts[0], m_pts[1], m_pts[2], m_pts[3]]]

    axis.add_collection3d(Poly3DCollection(sqr_verts,facecolors='black', linewidths=1, edgecolors='k', alpha=.5))
    axis.add_collection3d(Poly3DCollection(m_verts,facecolors='red', linewidths=1, edgecolors='r', alpha=1))

#Camera Intrinsic values from Project decription file
camera_intL = np.array([[423.27381306, 0., 341.34626532], 
					 [0., 421.27401756, 269.28542111], 
					 [0., 0., 1.]])

camera_intR = np.array([[420.91160482, 0, 352.16135589], 
					 [0, 418.72245958, 264.50726699], 
					 [0, 0, 1]])
dist_coeffL = np.array([-0.43394157423038077,0.26707717557547866,-0.00031144347020293427,0.0005638938101488364,-0.10970452266148858])
dist_coeffR = np.array([[-0.4145817681176909],[0.19961273246897668],[-0.00014832091141656534],[-0.0013686760437966467],[-0.05113584625015141]])

images = glob.glob(relative_path + 'left*.png')
positions, tvecs, rvecs, Rs = [], [], [], []

fig = plt.figure()
plt.axis('equal')
ax = fig.add_subplot(111, projection='3d')

for image in images:
	img = cv2.imread(image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
	parameters =  cv2.aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters) #corners are image points

	frame_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

	image_name = image.replace(relative_path, '')
	cv2.imwrite(image_name.replace('.png', '') + "_aruco.png", frame_markers)
	image_index = image_name.replace('left_', '').replace(".png", '')

	objp = np.float64([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0]])
	assert corners[0][0].shape[0] == objp.shape[0], 'points 3D and points 2D must have same number of vertices'
	ret, rvec, tvec = cv2.solvePnP(objp, corners[0][0], camera_intL, dist_coeffL)

	tvec = 5 * tvec 

	tvecs.append(tvec)
	rvecs.append(rvec)
    
    #Rotation vector --> Rotation matrix
	R, _ = cv2.Rodrigues(rvec) 
	Rs.append(R)
	position = -np.matrix(R).T * np.matrix(tvec)
	xs, ys, zs = position.item(0), position.item(1), position.item(2)

	ax.scatter(xs, ys, zs, c='b', marker='^', s = 0, zdir = 'z')
	ax.text(xs + 0.01, ys + 0.01, zs, image_index, zdir = 'y')

	positions.append(position)

	print(image_name)
	print("R: ", R.T)
	print("t: ", position )
	print("\n")

	plot_pyramid(ax, R.T, position)

plot_square(ax)
ax.set_xlim(-15, 10)
ax.set_ylim(10,-15)
ax.set_zlim(15,-30)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.tight_layout(rect=None, pad=0, h_pad=2)
plt.show()