import os
import random
import glob
from os import getcwd
import cv2
import numpy as np

def resize(frame_input, xdim=1920, ydim=1080):
    dim = (xdim, ydim)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

current_dir = getcwd() + '/'

# FIXME
in_dirname = "src_videos"
out_dirname = "split_images"
in_filename = []

for root, dirs, files in os.walk(in_dirname):
    for file in files:
        if '.MP4' in file:
            in_filename.append(file)

if not os.path.exists(out_dirname):
    os.mkdir(out_dirname)


for i, vid in enumerate(in_filename):
    cnt = 0
    print("Reading " + vid)
    cap = cv2.VideoCapture(vid)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = resize(frame, 640, 480)
            if not os.path.exists(out_dirname + '/' + in_filename[i]):
                os.mkdir(out_dirname + '/' + in_filename[i])
            cv2.imwrite(out_dirname + "/" + in_filename[i] + "/" + str(cnt) + ".jpg", frame)

            if cnt % 100 == 0:
                print("Reading frame # ", cnt)
            cnt += 1
        else:
            print("File grab failed")
            break
    cap.release()
