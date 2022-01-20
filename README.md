# hands_Jetson
2021-2 System Design Project for AI Applications (in SNU) Final Project

This the README file for the Final Project of System Design Project for AI.



A custom dataset was created through processing video files through scripts - (cropvideo.py, train_test_split.py, rotate.py, flip.py)

cropvideo.py

Calculates and extracts hand landmarks through the Google Mediapipe library,
Creates a bounding box using hand landmark coordinates,
Crops and resizes the files to match YOLO v4 input size [416x416]
Normalizes bounding box coordinates,
Removes frames either without a hand or with a fragment of one,
Classifies given video though its naming,
Saves four files - 

1. Video with hand landmarks and bounding box visualized - used for checking whether the various algorithms are working

2. Split and cropped video frame images - each labeled with its correct class, and index number within each class (ONE_1 ~ ONE_5000, TWO_1 ~ TWO_5000, etc.)

3. Split and cropped video frame images with bounding box visualized - used for algorithm checking

4. Text file containing class number and normalized bounding box coordinates for each image - for ease of processing later the names of each text file are kept the same as the image file

train_test_split.py

Selects training and validation images from the split frame set. This is done in order to 
1. randomize order and selection of frames from within a continous source
2. make sure that each class has an equal size, and no training or validation bias is introduced

rotate.py

Certain videos were mistakenly taken in a portrait aspect ratio.
These were rotated and turned into new landscape videos.

flip.py

Certain videos taken through the Galaxy S10 5G smartphone had its frames flipped horizontally,
and these were flipped back to original position to reduce underfitting within the model.
