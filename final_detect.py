import cv2
import dlib
import mediapipe as mp
import numpy as np
from imutils import face_utils
from keras.models import load_model


def Nose_Point(result, image_width, image_height):
    Nose_X = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
    Nose_Y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height

    return Nose_X, Nose_Y


def Compute_Neck(result, image_width, image_height):
    L_Shoulder_X = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
    R_Shoulder_X = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    Neck_X = (L_Shoulder_X + R_Shoulder_X) / float(2.0)

    R_Shoulder_Y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
    L_Shoulder_Y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
    Neck_Y = (L_Shoulder_Y + R_Shoulder_Y) / float(2.0)

    return Neck_X, Neck_Y


def Compute_Diff(Nose_X, Nose_Y, Neck_X, Neck_Y):
    Diff_X = Neck_X - Nose_X
    Diff_Y = Neck_Y - Nose_Y
    return Diff_X, Diff_Y


def Comput_Degree(Nose_X, Nose_Y, Neck_X, Neck_Y):
    AB = np.sqrt((Nose_X - Neck_X) ** 2 + (Nose_Y - Neck_Y) ** 2)
    AC = np.abs(Nose_X - Neck_X)
    cos = AC / AB
    degree = np.degrees(cos)
    return degree


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

pose = np.genfromtxt('data/pose_data.csv', delimiter=',')
pose = np.delete(pose, (0), axis=0)

feature = pose[:, :-1].astype(np.float32)
label = pose[:, -1].astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(feature, cv2.ml.ROW_SAMPLE, label)

model = load_model('models/eye_detecting.h5')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_eye/shape_predictor_68_face_landmarks.dat')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class_name = {0: "open", 1: "close"}

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Can't Read Camera...")

            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        result = pose.process(image)

        # compute degree and diff
        Nose_X, Nose_Y = Nose_Point(result, image_width, image_height)
        Neck_X, Neck_Y = Compute_Neck(result, image_width, image_height)
        Diff_X, Diff_Y = Compute_Diff(Nose_X, Nose_Y, Neck_X, Neck_Y)
        degree = Comput_Degree(Nose_X, Nose_Y, Neck_X, Neck_Y)
        data = np.array([degree, Diff_X, Diff_Y], dtype=np.float32)
        data = data.reshape(1, 3)

        ret, label, neighbours, dist = knn.findNearest(data, 5)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Flip the image horizontally for a selfie-view display.
        if label == 1:
            text = "Drowsy"
        else:
            text = "NonDrowsy"

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)

        for face in faces:
            shapes = predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)

            eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
            eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

            eye_img_l = cv2.resize(eye_img_l, dsize=(34, 26))
            eye_img_r = cv2.resize(eye_img_r, dsize=(34, 26))
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)

            # cv2.imshow("l",eye_img_l)
            # cv2.imshow("r", eye_img_r)

            eye_img_l3 = np.reshape(eye_img_l, (26, 34, 1))
            eye_img_r3 = np.reshape(eye_img_r, (26, 34, 1))
            r = model.predict(np.expand_dims(eye_img_r3, axis=0))
            l = model.predict(np.expand_dims(eye_img_l3, axis=0))

            if (int(r[0]) == 1 or int(l[0]) == 1):
                category = 0
            else:
                category = 1

            # if (int(r[0]) == 0 and int(l[0]) == 0):
            #     category = 1

            label_html = class_name[category]

        # create transparent overlay for bounding box

        cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, label_html, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if label == 1 or category == 1:
            cv2.putText(image, 'ALERT', (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('MediaPipe Pose', image)

        # Press esc to stop detect
        if cv2.waitKey(5) & 0xFF == 27:
            break

    print("Can't Open Camera...")
    exit()
    
cap.release()