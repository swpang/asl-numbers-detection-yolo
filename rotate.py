import cv2


root_dir = '/home/ai_competition10/Project/'


cap = cv2.VideoCapture(root_dir + 'TEST_3.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(root_dir + 'TEST_3_rot.mp4', fourcc, fps, (720, 1280))

cnt = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("ignoring frame")
        break
    print("writing frame ", cnt)
    cnt += 1
    image.flags.writeable = True
    cv2.flip(image, 0)
    cv2.flip(image, 1)
    out.write(image)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
cap.release()
cv2.destroyAllWindows()