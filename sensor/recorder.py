# This is a sample Python script.
# packages

import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io as sio
import time
import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def main(loc,img_num):

    pipeline_1 = rs.pipeline()
    config_1 = rs.config()

    config_1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    config_1.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

    # Start streaming from both cameras
    pipeline_1.start(config_1)
    for x in range(10):
        pipeline_1.wait_for_frames()
    depth_intrinsics=pipeline_1.wait_for_frames().profile.as_video_stream_profile().intrinsics
    img_counter = 0

    ## filters
    hole_filling = rs.hole_filling_filter()

    ## align tool

    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        time.sleep(0.5)
        frames_1 = pipeline_1.wait_for_frames()

        aligned_frames = align.process(frames_1)

        depth_frame_1 = aligned_frames.get_depth_frame()
        color_frame_1 = aligned_frames.get_color_frame()
        # color_frame_1 = frames_1.get_color_frame()
        # if not depth_frame_1:
        #     continue

        depth_frame_1 = hole_filling.process(depth_frame_1)

        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())



        save_loc_url = os.path.join(loc,"%03d.png"%(img_num))
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', color_image_1)
        cv2.waitKey(0)==ord('a')
        print("img saved : ", img_num)
        cv2.imwrite(save_loc_url,color_image_1)
        # cv2.imwrite("depth.png",depth_image_1,[cv2.IMWRITE_PNG_BILEVEL, 1])
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.5), cv2.COLORMAP_JET)
        # cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        # cv2.imshow('RealSense', depth_image_1)
        # cv2.waitKey(1)
        # plt.imshow(color_image_1)
        # plt.show()
        img_num +=1
        # time.sleep(1000)
    # except Exception as e:
    #     print(e)
    #     pass
if __name__ == '__main__':
    loc='/media/syh/ssd2/data/3d_reconstruction/rs_example/images'
    img_num=0
    main(loc,img_num)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
