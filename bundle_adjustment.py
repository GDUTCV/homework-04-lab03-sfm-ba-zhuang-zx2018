import numpy as np
import cv2


def compute_ba_residuals(parameters: np.ndarray, intrinsics: np.ndarray, num_cameras: int, points2d: np.ndarray,
                         camera_idxs: np.ndarray, points3d_idxs: np.ndarray) -> np.ndarray:
    """
    For each point2d in <points2d>, find its 3d point, reproject it back into the image and return the residual
    i.e. euclidean distance between the point2d and reprojected point.

    Args:
        parameters: list of camera parameters [r1, r2, r3, t1, t2, t3, ...] where r1, r2, r3 corresponds to the
                    Rodriguez vector. There are 6C + 3M parameters where C is the number of cameras
        intrinsics: camera intrinsics 3 x 3 array
        num_cameras: number of cameras, C
        points2d: N x 2 array of 2d points
        camera_idxs: camera_idxs[i] returns the index of the camera for points2d[i]
        points3d_idxs: points3d[points3d_idxs[i]] returns the 3d point corresponding to points2d[i]

    Returns:
        N residuals

    """
    num_camera_parameters = 6 * num_cameras
    camera_parameters = parameters[:num_camera_parameters]
    points3d = parameters[num_camera_parameters:]
    num_points3d = points3d.shape[0] // 3
    points3d = points3d.reshape(num_points3d, 3)

    camera_parameters = camera_parameters.reshape(num_cameras, 6)
    camera_rvecs = camera_parameters[:, :3]
    camera_tvecs = camera_parameters[:, 3:]

    extrinsics = []
    for rvec in camera_rvecs:
        rot_mtx, _ = cv2.Rodrigues(rvec)
        extrinsics.append(rot_mtx)
    extrinsics = np.array(extrinsics)  # C x 3 x 3
    extrinsics = np.concatenate([extrinsics, camera_tvecs.reshape(-1, 3, 1)], axis=2)  # C x 3 x 4

    residuals = np.zeros(shape=points2d.shape[0], dtype=float)
    """ 
    YOUR CODE HERE: 
    NOTE: DO NOT USE LOOPS 
    HINT: I used np.matmul; np.sum; np.sqrt; np.square, np.concatenate etc.
    """
    # 提取对应的3D点
    my3dp = points3d[points3d_idxs]

    # 将3D点齐次化
    home3dp = np.hstack((my3dp, np.ones((my3dp.shape[0], 1))))
    home3dp_T = home3dp.T  # 转置后的齐次3D点

    # 计算投影矩阵
    extrinsics_camera_idxs = extrinsics[camera_idxs]
    P = intrinsics @ extrinsics_camera_idxs

    # 使用对应的投影矩阵重新投影2D点
    calculated_2dp = np.einsum('ijk,ki->ij', P, home3dp_T)  # 使用爱因斯坦求和实现批量矩阵乘法
    calculated_2dp /= calculated_2dp[:, -1].reshape(-1, 1)  # 对齐次坐标进行归一化
    calculated_2dp = calculated_2dp[:, :-1]  # 去掉齐次坐标

    # 计算残差（欧几里得距离）
    residuals = np.linalg.norm(points2d - calculated_2dp, axis=1)

    """ END YOUR CODE HERE """
    return residuals
