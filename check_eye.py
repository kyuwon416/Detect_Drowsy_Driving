import cv2
import dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model


def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect

IMG_SIZE = (640, 480)

model = load_model('models/eye_detecting.h5')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_eye/shape_predictor_68_face_landmarks.dat')

label_html = 'Capturing...'

# initialze bounding box to empty
bbox = ''
count = 0
class_name = {0: "open", 1: "close"}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Can't Read Camera...")

        # If loading a video, use 'break' instead of 'continue'.
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=(34, 26))
        eye_img_r = cv2.resize(eye_img_r, dsize=(34, 26))
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        cv2.imshow("l",eye_img_l)
        cv2.imshow("r", eye_img_r)
        eye_img_l3 = np.reshape(eye_img_l, (26, 34, 1))
        eye_img_r3 = np.reshape(eye_img_r, (26, 34, 1))
        r = model.predict(np.expand_dims(eye_img_r3, axis=0))
        l = model.predict(np.expand_dims(eye_img_l3, axis=0))
        print(int(r[0]), " ", int(l[0]))
        if (int(r[0]) == 1 or int(l[0]) == 1):
            category = 0
        else:
            category = 1

        label_html = class_name[category]

    # create transparent overlay for bounding box
    bbox_array = np.zeros([480, 640, 4], dtype=np.uint8)

    # Press esc to stop detect
    if cv2.waitKey(1) == 27:
        break
    
print("Can't Open Camera...")
exit()