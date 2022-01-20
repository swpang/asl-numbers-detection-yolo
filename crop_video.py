import os
import random
import glob
from os import getcwd
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


RED = (0, 0, 255)

root_dir = '/home/ai_competition10/Project/'

video_dir_train = root_dir + "src_videos_palm/"
image_dir_train = root_dir + "data/"
video_dir_val = root_dir + "src_videos_palm_val/"
image_dir_val = root_dir + "data_val/"
outvideo_dir = root_dir + "append_videos_palm/"

def resize(frame_input, xdim=1920, ydim=1080):
    dim = (xdim, ydim)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)


def crop(frame_input, norm_x_center, norm_y_center, xdim, ydim, crop_x):
    x_center = xdim * norm_x_center
    y_center = ydim * norm_y_center

    crop_y_left = int(max(y_center - crop_x / 2, 0))
    crop_y_right = int(min(y_center + crop_x / 2, ydim))
    return frame_input[:, crop_y_left:crop_y_right], crop_y_left, crop_y_right


def get_idx(classname):
    switcher={
        'ONE': 0,
        'TWO': 1,
        'THREE': 2,
        'FOUR': 3,
        'FIVE': 4,
        'SIX': 5,
        'SEVEN': 6,
        'EIGHT': 7,
        'NINE': 8,
        'TEN': 9
    }
    key = classname
    return switcher.get(key, "invalid classname")


def get_video_files(in_dirname="src_videos"):
    in_filename = []
    for root, dirs, files in os.walk(in_dirname):
        for file in files:
            if '.MP4' or '.mp4' in file:
                in_filename.append(file)
    return in_filename


def get_photo_files(in_dirname):
    in_filename = []
    for root, dirs, files in os.walk(in_dirname):
        if len(dirs) == 0:
            for file in files:
                if '.jpg' in file:
                    in_filename.append(root + file)
    return in_filename


# create_landmark(video_dir, image_dir, outvideo_dir, src_vids_list, 640, 480)
def create_landmark(in_dirname, out_dirname, video_dirname, in_filename, xdim=1920, ydim=1080, crop_x=416, crop_y=416):
    if not os.path.exists(out_dirname):
        os.mkdir(out_dirname)
    if not os.path.exists(video_dirname):
        os.mkdir(video_dirname)

    counts = np.zeros((1, 10), dtype=int)

    for i, vid in enumerate(in_filename):
        temp = in_filename[i].split('_')
        temp1 = in_filename[i].split('.')
        classname = temp[0]
        idx = get_idx(classname)
        cnt = counts[0, idx]
        print("Reading " + vid)

        skipped = False

        cap = cv2.VideoCapture(in_dirname + '/' + vid)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_dirname + temp1[0] + '_' + "result.MP4", fourcc, fps, (ydim, xdim))

        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            max_num_hands=1,
            min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                cnt += 1
                success, image = cap.read()
                if not success:
                    print("Ignoring empty frame #{} Class: {}".format(cnt, classname))
                    break
                image = resize(image, ydim, xdim)
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    # Draw Landmarks to Input Videos
                    for hand_landmarks in results.multi_hand_landmarks:
                        bbox, norm_bbox = calc_bbox(hand_landmarks, xdim, ydim, padding_bbox=25)

                        crop_image, crop_bbox, norm_crop_bbox = crop_images(image, bbox, norm_bbox, xdim, ydim)
                        crop_image = resize(crop_image, crop_x, crop_y)

                        if not (norm_crop_bbox[0] - norm_crop_bbox[2] / 2 < 0.01 or norm_crop_bbox[0] + norm_crop_bbox[2] / 2 > 0.99
                            or norm_crop_bbox[1] - norm_crop_bbox[3] / 2 < 0.01 or norm_crop_bbox[1] + norm_crop_bbox[3] / 2 > 0.99):
                            if cnt % 6 == 0 or skipped == True:
                                cv2.imwrite(out_dirname + classname + '_' + str(int(cnt)) + ".jpg", crop_image)
                                # split video into frames
                                p0 = int((norm_crop_bbox[0] - norm_crop_bbox[2] / 2) * crop_y), int((norm_crop_bbox[1] - norm_crop_bbox[3] / 2) * crop_x)
                                p1 = int((norm_crop_bbox[0] + norm_crop_bbox[2] / 2) * crop_y), int((norm_crop_bbox[1] + norm_crop_bbox[3] / 2) * crop_x)
                                cv2.rectangle(crop_image, p0, p1, RED, 4)
                                cv2.imwrite(out_dirname + classname + '_' + str(int(cnt)) + "_bbox.jpg", crop_image)
                                create_labels(classname, out_dirname, cnt, norm_crop_bbox)
                                skipped = False
                        else:
                            skipped = True

                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        p0 = int((norm_bbox[0] - norm_bbox[2] / 2) * ydim), int((norm_bbox[1] - norm_bbox[3] / 2) * xdim)
                        p1 = int((norm_bbox[0] + norm_bbox[2] / 2) * ydim), int((norm_bbox[1] + norm_bbox[3] / 2) * xdim)
                        cv2.rectangle(image, p0, p1, RED, 4)

                    cv2.flip(image, 1)
                    out.write(image)

                    if cv2.waitKey(1) & 0xFF == ord('e'):
                        break
                    if cnt % 500 == 0:
                        print("Processing Class : {} frameNum : {} ....".format(classname, cnt))
        cap.release()
        cv2.destroyAllWindows()

        counts[0, idx] = cnt


def calc_bbox(landmarks, xdim, ydim, padding_bbox=10):
    norm_points = []
    for point in landmarks.landmark:
        norm_points.append([point.x, point.y])
    np_norm_points = np.array(norm_points, dtype=float)

    norm_x_max = min(np.max(np_norm_points[:,1]) + padding_bbox / xdim, 1)
    norm_x_min = max(np.min(np_norm_points[:,1]) - padding_bbox / xdim, 0)
    norm_y_max = min(np.max(np_norm_points[:,0]) + padding_bbox / ydim, 1)
    norm_y_min = max(np.min(np_norm_points[:,0]) - padding_bbox / ydim, 0)

    norm_x_center = (norm_x_min + norm_x_max) / 2.0
    norm_y_center = (norm_y_min + norm_y_max) / 2.0
    norm_width = norm_y_max - norm_y_min
    norm_height = norm_x_max - norm_x_min

    x_center = norm_x_center * xdim
    y_center = norm_y_center * ydim
    width = norm_width * ydim
    height = norm_height * xdim

    return [y_center, x_center, width, height], [norm_y_center, norm_x_center, norm_width, norm_height]


# def crop(frame_input, norm_x_center, norm_y_center, xdim, ydim, crop_x, crop_y):
def crop_images(frame, bbox, norm_bbox, xdim, ydim):
    new_img, crop_y_left, crop_y_right = crop(frame, norm_bbox[1], norm_bbox[0], xdim, ydim, xdim)

    new_y_left = max(bbox[0] - bbox[2] / 2 - crop_y_left, 0)
    new_y_right = min(bbox[0] + bbox[2] / 2 - crop_y_left, crop_y_right - crop_y_left)

    new_y_center = (new_y_left + new_y_right) / 2
    new_width = new_y_right - new_y_left

    new_norm_y_center = new_y_center / (crop_y_right - crop_y_left)
    new_norm_width = new_width / (crop_y_right - crop_y_left)

    return new_img, [new_y_center, bbox[1], new_width, bbox[3]], [new_norm_y_center, norm_bbox[1], new_norm_width, norm_bbox[3]]


def create_labels(classname, out_dirname, frameNum, norm_bbox):
    txt_outfile = open(out_dirname + classname + '_' + str(int(frameNum)) + '.txt', "w")

    txt_outfile.write(
        str(get_idx(classname))
        + ' ' + str(norm_bbox[0])
        + ' ' + str(norm_bbox[1])
        + ' ' + str(norm_bbox[2])
        + ' ' + str(norm_bbox[3])
    )


def main():
    src_vids_list_train = get_video_files(video_dir_train)
    src_vids_list_val = get_video_files(video_dir_val)
    create_landmark(video_dir_train, image_dir_train, outvideo_dir, src_vids_list_train, 480, 854)
    create_landmark(video_dir_val, image_dir_val, outvideo_dir, src_vids_list_val, 480, 854)
    print("Complete")


if __name__ == '__main__':
    main()
