import pandas as pd
import numpy as np
import mediapipe as mp
import cv2
import os

def Point_DataFrame(results, label):
    point_list = []
    if label == 0 or label == "notdrowsy":
        label = 0
    
    elif label == 1 or label == "drowsy":
        label = 1
    
    else:
        label = -1
    
    for result in results:
        temp = []
        Neck_X, Neck_Y = Compute_Neck(result, image_width, image_height)
        Nose_X, Nose_Y = Nose_Point(result, image_width, image_height)
        R_Shoulder_X, R_Shoulder_Y,L_Shoulder_X, L_Shoulder_Y = Shoulder_Point(result, image_width, image_height)
        temp.append(Neck_X)
        temp.append(Neck_Y)
        temp.append(Nose_X)
        temp.append(Nose_Y)
        temp.append(R_Shoulder_X)
        temp.append(R_Shoulder_Y)
        temp.append(L_Shoulder_X)
        temp.append(L_Shoulder_Y)
        temp.append(label)
        point_list.append(temp)
    
    dataframe = pd.DataFrame(
        columns = ["Neck_X", "Neck_Y", "Nose_X", "Nose_Y", "R_Shoulder_X", "R_Shoulder_Y","L_Shoulder_X", "L_Shoulder_Y", "label"],
        data = point_list
        )
    
    return dataframe

def Shoulder_Point(result, image_width, image_height) :
    mp_pose = mp.solutions.pose

    L_Shoulder_X = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
    R_Shoulder_X = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    
    
    R_Shoulder_Y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
    L_Shoulder_Y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
    
    return R_Shoulder_X, R_Shoulder_Y,L_Shoulder_X, L_Shoulder_Y

def Compute_Neck(result, image_width, image_height) :
    mp_pose = mp.solutions.pose

    L_Shoulder_X = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
    R_Shoulder_X = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    Neck_X = (L_Shoulder_X + R_Shoulder_X) / float(2.0)
    
    R_Shoulder_Y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
    L_Shoulder_Y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
    Neck_Y = (L_Shoulder_Y + R_Shoulder_Y) / float(2.0)

def Nose_Point(result, image_width, image_height) :
    mp_pose = mp.solutions.pose

    Nose_X = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
    Nose_Y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height

    return Nose_X, Nose_Y

def Point_by_MediaPipe(image_path, train_test, label):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
    if train_test == "train":
        train_test = "train/"
    else:
        train_test = "test/"
    
    if label == "drowsy":
        label = "drowsy/"
    else:
        label = "notdrowsy/"
        
    # For static images:
    IMAGE_FILES = image_path
    results = []
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            file = "pose/" + train_test + label + file
            print(file)
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            results.append(pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

            if not results[idx].pose_landmarks:
                continue
                
            if idx % 10 == 0:
                print("전체 이미지 : %d / 완료한 이미지 %d" % (len(IMAGE_FILES), idx))
    print("---완료---")
    return results, image_height, image_width

def Compute_Diff(dataframe):
    dataframe["Diff_X"] = dataframe["Neck_X"] - dataframe["Nose_X"]
    dataframe["Diff_Y"] = dataframe["Neck_Y"] - dataframe["Nose_Y"]
    return dataframe

def Comput_Degree(dataframe):
    AB = np.sqrt((dataframe["Nose_X"] - dataframe["Neck_X"]) ** 2 
              + (dataframe["Nose_Y"] - dataframe["Neck_Y"]) ** 2)
    AC = np.abs(dataframe["Nose_X"] - dataframe["Neck_X"])
    cos = AC / AB
    degree = np.degrees(cos)
    dataframe.insert(0, "degree", degree, True)
    return dataframe


PATH_TRAIN_NOTDROWSY_LIST = os.listdir("pose/train/notdrowsy/")
PATH_TRAIN_DROWSY_LIST  = os.listdir("pose/train/drowsy/")

drowsy_results, image_height, image_width = Point_by_MediaPipe(PATH_TRAIN_DROWSY_LIST, "train", "drowsy")
notdrowsy_results, image_height, image_width = Point_by_MediaPipe(PATH_TRAIN_NOTDROWSY_LIST, "train", "notdrowsy")

train_data_drowsy = Point_DataFrame(drowsy_results, "drowsy")
train_data_notdrowsy = Point_DataFrame(notdrowsy_results, "notdrowsy")

train_data_drowsy = Compute_Diff(train_data_drowsy)
train_data_notdrowsy = Compute_Diff(train_data_notdrowsy)

train_data_drowsy = Comput_Degree(train_data_drowsy)
train_data_notdrowsy = Comput_Degree(train_data_notdrowsy)

train_data_drowsy = train_data_drowsy.drop(train_data_drowsy.columns[[1,2,3,4,5,6,7,8]], axis = 1)
train_data_notdrowsy = train_data_notdrowsy.drop(train_data_notdrowsy.columns[[1,2,3,4,5,6,7,8]], axis = 1)

train_data_drowsy = train_data_drowsy[['degree', 'Diff_X', 'Diff_Y', 'label']]
train_data_notdrowsy = train_data_notdrowsy[['degree', 'Diff_X', 'Diff_Y', 'label']]

DataSet = pd.concat([train_data_drowsy, train_data_notdrowsy], ignore_index= True)

DataSet.to_csv("data/Pose_Dataset/pose_data_.csv", index = False)