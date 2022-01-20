import argparse
import numpy
import cv2
import os

root_dir = '/home/ai_competition10/Project/'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_vid', '-i', dest='input_vid', help='path of the input video')
parser.add_argument('--output_vid', '-o', dest='output_vid', help='path of the output video')
args = parser.parse_args()


def resize(frame_input, xdim=1920, ydim=1080):
    dim = (xdim, ydim)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture(args.input_vid)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# out = cv2.VideoWriter(args.output_vid, fourcc, fps, (int(height), int(width)))
out = cv2.VideoWriter(args.output_vid, fourcc, fps, (1920, 1080))

cnt = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("ignoring frame")
        break
    print("writing frame #", cnt)
    cnt += 1

    hei, wid, _ = image.shape
    if hei > wid:
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)

    image = cv2.flip(image, 1)
    image = resize(image, 1920, 1080)
    out.write(image)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
cap.release()
cv2.destroyAllWindows()
