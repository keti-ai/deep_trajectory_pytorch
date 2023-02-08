# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io as sio


def main():

    meta = sio.loadmat("meta_own.mat")
    print(meta.keys())
    for i in meta.keys():
        print(i)
        print(meta[i])
    print(meta['poses'].shape)
    # raise KeyboardInterrupt
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()

    # config_1.enable_device('802212060621')
    config_1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    # config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.bgr8, 30)
    config_1.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

    # Start streaming from both cameras
    pipeline_1.start(config_1)
    for x in range(5):
        pipeline_1.wait_for_frames()
    depth_intrinsics=pipeline_1.wait_for_frames().profile.as_video_stream_profile().intrinsics
    print(depth_intrinsics)
    print(depth_intrinsics.fx)
    print(depth_intrinsics.fy)
    print(depth_intrinsics.ppx)
    print(depth_intrinsics.ppy)
    print(depth_intrinsics.coeffs)

    #
    # meta['intrinsic_matrix'][0,0]=depth_intrinsics.fx
    # meta['intrinsic_matrix'][1,1]=depth_intrinsics.fy
    # meta['intrinsic_matrix'][0,2]=depth_intrinsics.ppx
    # meta['intrinsic_matrix'][1,2]=depth_intrinsics.ppy
    # meta['factor_depth']=750

    meta['factor_depth']=1200
    sio.savemat("meta_own.mat",meta)
    meta_own=sio.loadmat("meta_own.mat")
    print(meta_own)
    # raise KeyboardInterrupt
    img_counter = 0
    # try:

    ## filters
    hole_filling = rs.hole_filling_filter()

    ## align tool

    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        # Camera 1
        # Wait for a coherent pair of frames: depth and color
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




        cv2.imwrite("color.png",color_image_1)
        cv2.imwrite("depth.png",depth_image_1,[cv2.IMWRITE_PNG_BILEVEL, 1])
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.5), cv2.COLORMAP_JET)
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', depth_image_1)
        cv2.waitKey(1)
        # plt.imshow(color_image_1)
        # plt.show()
        import time
        # time.sleep(1000)
    # except Exception as e:
    #     print(e)
    #     pass
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
