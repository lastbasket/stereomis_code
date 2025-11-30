import sys
sys.path.append('../')
import os
import torch
import numpy as np
from tqdm import tqdm
import cv2


def disparity_to_img3d(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Convert disparity to 3D image
    
    Convert disparity to 3D image format, similar to what is used to store
    ground truth information in scared. The resulting 3D image is expressed in
    the same frame of reference with the disparity, thus it cannot directly used
    to create 3D images suitable for evaluation on the provided sequence. The 
    unprojection is done using the Q matrix computed during the stereo 
    calibration and rectification phase.

    Args:
        disparity (np.ndarray): HxW disparity map float array
        Q (np.ndarray): Q matrix computed during stereo calibration and 
        rectification phase.

    Returns:
        np.ndarray: HxWx3 img3d output array. Each pixel location (u,v) encodes
        the 3D coordinates of the 3D point that projects to u,v.
    """
    assert disparity.dtype == np.float32
    
    disparity = np.nan_to_num(disparity)
    valid = disparity >= 0
    # print(disparity.shape)
    img3d = cv2.reprojectImageTo3D(disparity, Q)
    # print(img3d.shape)
    img3d[~valid] = np.nan
    
    return img3d

def disparity_to_depthmap(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """ Convert disparity to depthmap
    
    Converts a disparity image to depthmap using the provided Q matrix copmuted
    during the stereo calibration and rectification phase. Q is used to 
    unproject the disparity image and constact a 3D image. This function returns
    the 3rd channel of this 3D image which is defined as the depthmap. The output
    of this function can not be used evaluate on the original SCARED ground truth
    as the former is expressed in a stereo rectified frame of reference while the
    former is express in the original frame of reference including distortions.

    Args:
        disparity (np.ndarray): HxW disparity map float array
        Q (np.ndarray):  Q matrix computed during stereo calibration and 
        rectification phase.

    Returns:
        np.ndarray: HxW depthmap float array represented in the same frame of
        reference as the input disparity image.
    """
    img3d = disparity_to_img3d(disparity, Q)
    return img3d_to_depthmap(img3d)

def img3d_to_depthmap(img3d: np.ndarray) -> np.ndarray:
    """Converts 3D image to depthmap
    
    Convertion is simple because the depthmap is the 3rd channel of a 3D image.

    Args:
        img3d (np.ndarray): HxWx3 array, each pixel location (u,v) encodes
        the 3D coordinates of the a point that projects to u,v.


    Returns:
        np.ndarray: HxW float depthmap
    """
    # print(img3d[~np.isnan(img3d)].shape)
    # print(img3d[~np.isnan(img3d)].min(), img3d[~np.isnan(img3d)].max())
    result = img3d[:, :, 2].copy()
    # print(result[~np.isnan(result)].min(), result[~np.isnan(result)].max())
    return result

def depth_map(imgL, imgR, q):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 7  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    gray_left = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    gray_right = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=16,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=-1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=100,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 8000
    sigma = 1.2

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(gray_left, gray_right).astype(np.float32)/16
    dispr = right_matcher.compute(gray_left, gray_right).astype(np.float32)/16

    
    f_displ = wls_filter.filter(displ, gray_left, None, dispr)
    f_displ = cv2.bilateralFilter(f_displ, 5, 75, 75)
    
    depthl = disparity_to_depthmap(f_displ, q)
    # print(depthl.shape, depthl[~np.isnan(depthl)].min(), depthl[~np.isnan(depthl)].max())
    depthl[np.isnan(depthl)] = 0
    zero_mask = (depthl != 0)
    close_depth = np.percentile(depthl[depthl!=0], 5)
    inf_depth = np.percentile(depthl, 95)
    depthl = np.clip(depthl, close_depth, inf_depth)
    # print(depthl.min(), depthl.max())
    
    dep_visl = None
    dep_visl = cv2.normalize(src=depthl, dst=dep_visl, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    # print(dep_visl.min(), dep_visl.max(), dep_visl.mean())

    disp_visl = None
    disp_visl = cv2.normalize(src=f_displ, dst=disp_visl, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)


    return depthl, dep_visl, displ, dispr, disp_visl