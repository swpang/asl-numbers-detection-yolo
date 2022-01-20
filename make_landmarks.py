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


def resize(frame_input, xdim=1920, ydim=1080):
    dim = (xdim, ydim)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)


def get_files(in_dirname="src_videos"):
    in_filename = []
    for root, dirs, files in os.walk(in_dirname):
        for file in files:
            if '.MP4' in file:
                in_filename.append(file)
    return in_filename


def split_frames(in_dirname, in_filename, out_dirname, xdim=1920, ydim=1080):
    if not os.path.exists(out_dirname):
        os.mkdir(out_dirname)

    for i, vid in enumerate(in_filename):
        cnt = 0
        print("Reading " + vid)
        cap = cv2.VideoCapture(in_dirname + '/' + vid)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = resize(frame, xdim, ydim)
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


def create_landmark(in_dirname, in_filename, out_dirname, xdim=1920, ydim=1080):
    if not os.path.exists(out_dirname):
        os.mkdir(out_dirname)

    for i, vid in enumerate(in_filename):
        if not os.path.exists(out_dirname + '/' + in_filename[i]):
            os.mkdir(out_dirname + '/' + in_filename[i])

        cnt = 0
        print("Reading ", vid)
        cap = cv2.VideoCapture(in_dirname + '/' + vid)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_dirname + '/' + in_filename[i] + '/' + "result.MP4", fourcc, fps, (xdim, ydim))

        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            max_num_hands=1,
            min_tracking_confidence=0.5) as hands:
            frameNum = 0
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame #", frameNum)
                    break
                image.flags.writeable = False

                image = resize(image, xdim, ydim)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    # Draw Landmarks to Input Videos
                    image_height, image_width, _ = image.shape
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = [
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
                            ],
                            [
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width,
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height
                            ],
                        ]

                        create_npz(in_filename[i], frameNum, out_dirname, landmarks)

                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    cv2.flip(image, 1)
                    out.write(image)

                    if cv2.waitKey(1) & 0xFF == ord('e'):
                        break
                    frameNum += 1
                    if frameNum % 500 == 0:
                        print("frameNum : ", frameNum)
        cap.release()
        cv2.destroyAllWindows()


def create_npz(vid_name, frameNum, out_dir, landmarks):
    if not len(landmarks) == 0:
        np_temp = np.array(landmarks, dtype=float)
        np.savez_compressed(out_dir + '/' + vid_name + '/' + str(frameNum) + '.npz', np_temp)
    else:
        print("No landmarks detected in", vid_name)
        print("Frame Number : ", frameNum)


def main():
    current_dir = getcwd() + '/'
    root_dir = '/tools/home/ai_competition10/Project/MobileNetV3small_gesture/'

    input_dir = root_dir + "src_vids/"
    output_dir = root_dir + "landmarks/"

    src_vids_list = get_files(input_dir)
    # split_frames(input_dir, src_vids_list, output_dir, 640, 480)
    create_landmark(input_dir, src_vids_list, output_dir, 640, 480)

    print("Complete")


if __name__ == '__main__':
    main()
