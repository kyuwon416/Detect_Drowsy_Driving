import cv2
import mediapipe as mp
import numpy as np
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


pose = np.genfromtxt('data/pose_data.csv', delimiter=',')
pose = np.delete(pose, (0), axis=0)

feature = pose[:, :-1].astype(np.float32)
label = pose[:, -1].astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(feature, cv2.ml.ROW_SAMPLE, label)

model = load_model('models/eye_detecting.h5')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

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

        # create transparent overlay for bounding box

        cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Pose', image)

        # Press esc to stop detect
        if cv2.waitKey(5) & 0xFF == 27:
            break

    print("Can't Open Camera...")
    exit()
cap.release()