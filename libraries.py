import os
import cv2
import shutil
import time
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from imutils import build_montages
import face_recognition_models
from tqdm import tqdm
from pyPiper import Node, Pipeline

class ResizeUtils:
    @staticmethod
    def rescale_by_height(image, target_height, method=cv2.INTER_LANCZOS4):
        w = int(round(target_height * image.shape[1] / image.shape[0]))
        return cv2.resize(image, (w, target_height), interpolation=method)

    @staticmethod
    def rescale_by_width(image, target_width, method=cv2.INTER_LANCZOS4):
        h = int(round(target_width * image.shape[0] / image.shape[1]))
        return cv2.resize(image, (target_width, h), interpolation=method)

class FramesGenerator:
    def __init__(self, video_source):
        self.video_source = video_source

    def auto_resize(self, frame):
        resize_utils = ResizeUtils()
        height, width, _ = frame.shape
        if height > 500:
            frame = resize_utils.rescale_by_height(frame, 500)
        if width > 700:
            frame = resize_utils.rescale_by_width(frame, 700)
        return frame

    def generate_frames(self, output_directory):
        cap = cv2.VideoCapture(self.video_source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(output_directory, exist_ok=True)
        frame_count, fps_counter = 0, 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if fps_counter > fps:
                fps_counter = 0
                frame = self.auto_resize(frame)
                cv2.imwrite(os.path.join(output_directory, f"frame_{frame_count}.jpg"), frame)
            fps_counter += 1
            frame_count += 1
        cap.release()

class FaceEncoder(Node):
    def setup(self, detection_method='cnn'):
        self.detection_method = detection_method

    def run(self, data):
        image_path = data['imagePath']
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition_models.face_locations(rgb, model=self.detection_method)
        encodings = face_recognition_models.face_encodings(rgb, boxes)
        self.emit({'id': data['id'], 'encodings': [{'imagePath': image_path, 'loc': box, 'encoding': enc} for box, enc in zip(boxes, encodings)]})

class PicklesListCollator:
    def __init__(self, input_directory):
        self.input_directory = input_directory

    def generate_pickle(self, output_file):
        data = []
        for file in os.listdir(self.input_directory):
            if file.endswith('.pickle'):
                with open(os.path.join(self.input_directory, file), 'rb') as f:
                    data.extend(pickle.load(f))
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

class FaceClusterUtility:
    def __init__(self, encoding_file):
        self.encoding_file = encoding_file

    def cluster(self):
        with open(self.encoding_file, 'rb') as f:
            data = pickle.load(f)
        encodings = [d['encoding'] for d in data]
        clt = DBSCAN(eps=0.5, metric='euclidean', n_jobs=-1)
        clt.fit(encodings)
        return clt.labels_

class FaceImageGenerator:
    def __init__(self, encoding_file):
        self.encoding_file = encoding_file

    def generate_images(self, labels, output_directory='ClusteredFaces', montage_folder='Montage'):
        os.makedirs(output_directory, exist_ok=True)
        montage_path = os.path.join(output_directory, montage_folder)
        os.makedirs(montage_path, exist_ok=True)
        with open(self.encoding_file, 'rb') as f:
            data = pickle.load(f)
        for label in np.unique(labels):
            label_folder = os.path.join(output_directory, f"Face_{label}")
            os.makedirs(label_folder, exist_ok=True)
            idxs = np.where(labels == label)[0]
            images = [cv2.imread(data[i]['imagePath']) for i in idxs]
            montage = build_montages(images, (96, 120), (5, 5))[0]
            cv2.imwrite(os.path.join(montage_path, f"Face_{label}.jpg"), montage)
