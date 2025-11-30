import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from stereomis.rectification import StereoRectifier
import torch
from glob import glob
import tqdm
from depth_utils import depth_map
import matplotlib


cmap = matplotlib.cm.get_cmap('Spectral_r')


def process_one(dir_name, start=0, end=1000000):
    data_dir = dir_name
    video  = glob(os.path.join(data_dir,'*.mp4'))[0]

    calib_file = os.path.join(data_dir, 'StereoCalibration.ini')
    capture = cv2.VideoCapture(video)

    rect = StereoRectifier(calib_file, img_size_new=None)
    calib = rect.get_rectified_calib()
    count = 0
    folder_left = 'images_left_rect'
    folder_right = 'images_right_rect'
    folder_depth = 'depth'
    folder_depth_vis = 'depth_vis'
    # disp_path = 'disparity_test'
    
    os.makedirs(os.path.join(data_dir, folder_left), exist_ok=True)
    os.makedirs(os.path.join(data_dir, folder_right), exist_ok=True)
    
    # os.makedirs(os.path.join(data_dir, disp_path), exist_ok=True)
    os.makedirs(os.path.join(data_dir, folder_depth), exist_ok=True)
    os.makedirs(os.path.join(data_dir, folder_depth_vis), exist_ok=True)

    # split = [16400, 16699]
    # split = [6800, 6899]
    # split = [8350, 8449]
    # split = [7379, 7499]
    split = [start, end]
    
    pbar = tqdm.tqdm()
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        count_tring = str(count).rjust(6, '0')

        if (count>=split[0]) and (count <= split[1]):
            if count %2 == 0:
                count += 1
                continue
            left_path = os.path.join(data_dir, f'{folder_left}/{count_tring}.png')
            # if os.path.isfile(left_path):
            #     continue
            h, w, c = frame.shape
            left = torch.from_numpy(cv2.cvtColor(frame[0:h//2, :, :], cv2.COLOR_BGR2RGB)).permute(2,0,1).float()
            right = torch.from_numpy(cv2.cvtColor(frame[h//2:, :, :], cv2.COLOR_BGR2RGB)).permute(2,0,1).float()
            left_calib, right_calib = rect(left, right)
            # print(calib['q'])
            depthl, dep_visl, displ, dispr, disp_visl = \
                depth_map(left_calib.permute(1,2,0).cpu().numpy(), \
                    right_calib.permute(1,2,0).cpu().numpy(), calib['q'])
            left_path = os.path.join(data_dir, f'{folder_left}/{count_tring}.png')
            right_path = os.path.join(data_dir, f'{folder_right}/{count_tring}.png')
            # dis_path = os.path.join(data_dir, f'disparity/{count_tring}.png')
            dep_path = os.path.join(data_dir, f'{folder_depth}/{count_tring}.npy')
            dep_vis_path = os.path.join(data_dir, f'{folder_depth_vis}/{count_tring}.png')
            
            depth = depthl
            # depth = cv2.resize(depth, (w//2, h//4))
            close_depth = np.percentile(depth[depth!=0], 5)
            inf_depth = np.percentile(depth, 95)
            depth[depth==close_depth] = 0

            np.save(dep_path, depth)
            
            # depth_vis = (cmap(dep_visl)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            depth_vis = depth/depth.max()*255.0
            depth_vis = depth_vis.astype(np.uint8)

            left_calib = left_calib.permute(1,2,0).numpy()
            right_calib = right_calib.permute(1,2,0).numpy()
            # left_calib = cv2.resize(left_calib, ((w//2, h//4)))
            cv2.imwrite(left_path, cv2.cvtColor(left_calib, cv2.COLOR_RGB2BGR).astype(np.uint8))
            cv2.imwrite(right_path, cv2.cvtColor(right_calib, cv2.COLOR_RGB2BGR).astype(np.uint8))
            cv2.imwrite(dep_vis_path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR).astype(np.uint8))
            print(left_path)
            
            pbar.update(1)
            
        count += 1
        
    
folder_list = os.path.join('StereoMIS_0_0_1', "P2_0")

process_one(folder_list, start=3700, end=3900)
