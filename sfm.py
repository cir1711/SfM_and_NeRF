import os
import cv2
import numpy             as np
import matplotlib.pyplot as plt
from os             import listdir
from os.path        import join as pathjoin
from scipy.optimize import least_squares
from scipy.sparse   import lil_matrix
from tqdm           import tqdm
from plyfile        import PlyData

def load_dataset(path):
    
    images = [cv2.cvtColor(cv2.imread(pathjoin(path, img)), cv2.COLOR_BGR2RGB) for img in sorted(listdir(path))]
    h, w, _ = images[0].shape
    focal = max(w,h) * 1.05
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
    return images, K

def find_descriptors_keypoints_all(images):
    sift = cv2.SIFT_create()
    return zip(*[sift.detectAndCompute(img, None) for img in images])

def match_features(kpi, desi, kpj, desj):
    bf = cv2.BFMatcher()
    matches = [m for m,n in bf.knnMatch(desi, desj, k=2) if m.distance < 0.7 * n.distance] 
    idx1 = np.array([m.queryIdx for m in matches])
    idx2 = np.array([m.trainIdx for m in matches])
    img1pts = np.float32([ kpi[m.queryIdx].pt for m in matches ])
    img2pts = np.float32([ kpj[m.trainIdx].pt for m in matches ])
    return img1pts, img2pts, idx1, idx2

def pose_estimation(img1pts, img2pts, idx1, idx2, K):
    E, mask = cv2.findEssentialMat(img1pts, img2pts, K, method=cv2.RANSAC, prob=0.999, threshold=1)
    img1pts, img2pts = img1pts[mask.ravel() == 1], img2pts[mask.ravel() == 1]
    idx1, idx2       =    idx1[mask.ravel() == 1],    idx2[mask.ravel() == 1]
    _, R, t, mask = cv2.recoverPose(E, img1pts, img2pts, K)
    return img1pts, img2pts, idx1, idx2, R, t

def triangulate_two_view(pose0, pose1, img1pts, img2pts):
    pts3D = cv2.triangulatePoints(pose0, pose1, img1pts.T, img2pts.T)
    pts3D = pts3D / pts3D[3]
    return pts3D

def match_2D_to_3D(cloud, colors, pts3D, kp, idx1, idx2, index, img, match2d3d): 
    pts3D = cv2.convertPointsFromHomogeneous(pts3D.T)
    pts3D = pts3D[:, 0, :]
    index_2d = idx2[np.where(match2d3d[index][idx1] == -1)]
    for w, i in enumerate(idx1): 
        if match2d3d[index][i] == -1:
            cloud.append(pts3D[w])
            colors.append(img[int(kp[i].pt[1]), int(kp[i].pt[0])])
            match2d3d[index][i] = len(cloud) - 1
        match2d3d[index+1][idx2[w]] = match2d3d[index][i]
    pts3D_index = match2d3d[index+1][index_2d]
    return cloud, colors, pts3D_index, index_2d, match2d3d

def jac_sparsity(point3d_ind, x):
    A = lil_matrix((len(point3d_ind)*2, len(x)), dtype=int)
    A[np.arange(len(point3d_ind)*2), :6] = 1
    for i in range(3):
        A[np.arange(len(point3d_ind))*2, 6 + np.arange(len(point3d_ind))*3 + i] = 1
        A[np.arange(len(point3d_ind))*2 + 1, 6 + np.arange(len(point3d_ind))*3 + i] = 1
    return A

def bundle_adjustment(cloud, pts3D_index, index_2d, kp, K, R2, t2):
    x = np.hstack((
        cv2.Rodrigues(R2)[0].ravel(),
        t2, 
        np.array(cloud)[pts3D_index].ravel()
    ))
    A = jac_sparsity(pts3D_index, x)
    kp = np.array([_kp.pt for _kp in kp])
    res = least_squares(calc_reprojection_err, x, jac_sparsity=A, x_scale='jac', ftol=1e-8, args=(K, kp[index_2d]))
    R, t, point_3D = cv2.Rodrigues(res.x[:3])[0], res.x[3:6], res.x[6:].reshape((len(pts3D_index), 3))
    for i, j in enumerate(pts3D_index): cloud[j] = point_3D[i]
    return cloud, R, t

def calc_reprojection_err(x, K, point_2D):
    R, t, point_3D = x[:3], x[3:6], x[6:].reshape((len(point_2D), 3))
    reprojected_point, _ = cv2.projectPoints(point_3D, R, t, K, distCoeffs=None)
    reprojected_point = reprojected_point[:, 0, :]
    return (point_2D - reprojected_point).ravel()

def to_ply(img_dir, point_cloud, colors):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)

    verts = np.hstack([out_points, out_colors])
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    ply_header = "ply\nformat ascii 1.0\nelement vertex %(vert_num)d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\ncomment pati\nend_header\n"
    ply_result = pathjoin(img_dir,'sparse.ply')
    with open(ply_result, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')
    ply_file = PlyData.read(ply_result)
    ply_file.text = False
    ply_file.write(ply_result)
    
def run(path):
    images, K = load_dataset(path)
    total_images = len(images)
    cloud, colors = [], []
    keypoints, descriptors = find_descriptors_keypoints_all(images)
    match2d3d = [np.ones((len(kp), ) , dtype='int32')*-1 for kp in keypoints]

    #Start incremental structure from motion first image pair
    img1pts, img2pts, idx1, idx2 = match_features(keypoints[0], descriptors[0], keypoints[1], descriptors[1])
    
    #Get R and t
    img1pts, img2pts, idx1, idx2, R, t = pose_estimation(img1pts, img2pts, idx1, idx2, K)

    #Proyections matrices
    trans_mat0 = np.eye(3,4)
    trans_mat1 = np.hstack((R, t))
    pose0, pose1 = np.dot(K, trans_mat0), np.dot(K, trans_mat1)

    #Get 3D point
    pts3D = triangulate_two_view(pose0, pose1, img1pts, img2pts)
    
    #match 2d points and 3d points
    obj = match_2D_to_3D(cloud, colors, pts3D, keypoints[0], idx1, idx2, 0, images[0], match2d3d)
    cloud, colors, pts3D_index, index_2d, match2d3d = obj
    
    #bundle adjustment
    cloud, R2, t2 = bundle_adjustment(cloud, pts3D_index, index_2d, keypoints[1], K, trans_mat1[:3,:3], t.ravel())
    trans_mat0 = np.hstack((R2, t2.reshape((3,1))))

    for i in tqdm(range(1, total_images-1)):
        img1pts, img2pts, idx1, idx2 = match_features(keypoints[i], descriptors[i], keypoints[i+1], descriptors[i+1])
        img1pts, img2pts, idx1, idx2, R, t = pose_estimation(img1pts, img2pts, idx1, idx2, K)

        match = np.int32(np.where(match2d3d[i][idx1] != -1)[0])
        if len(match) < 8: continue
        kp = np.array([_kp.pt for _kp in keypoints[i+1]])
        _, rvecs, t, _ = cv2.solvePnPRansac(np.float32(cloud)[match2d3d[i][idx1[match]]], kp[idx2[match]], K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
        R2, _ = cv2.Rodrigues(rvecs)
        trans_mat1 = np.hstack((R2, t))
        pose0, pose1 = np.dot(K, trans_mat0), np.dot(K, trans_mat1)
        pts3D = triangulate_two_view(pose0, pose1, img1pts, img2pts)
       
        #match 2d points and 3d points
        obj = match_2D_to_3D(cloud, colors, pts3D, keypoints[i], idx1, idx2, i, images[i], match2d3d)
        cloud, colors, pts3D_index, index_2d, match2d3d = obj

        #bundle adjustment
        cloud, R2, t2 = bundle_adjustment(cloud, pts3D_index, index_2d, keypoints[i+1], K, trans_mat1[:3,:3], t.ravel())
        trans_mat0 = np.hstack((R2, t2.reshape((3,1))))
    
    to_ply(path, np.array(cloud), np.array(colors))
    """aux = np.array(cloud)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = aux[:, 0], aux[:, 2], -aux[:,1]
    ax.scatter(x, y, z, c=np.array(colors)/255, marker='o')
    plt.show()"""

"""
run('fernn/')
run('fortress/')
run('horns/')"""
run('monument/')
""""""
run('pine_cone/')
run('trex/')
run('vasedeck/')
run('vasedeck2/')
run('south-building/')