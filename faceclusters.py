import os
import cv2
import pickle
import shutil
import numpy as np
from sklearn.cluster import DBSCAN
from imutils import build_montages

def load_encodings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Encoding file {file_path} not found")
    with open(file_path, "rb") as f:
        return np.array(pickle.load(f))

def cluster_faces(encodings, eps=0.5, jobs=-1):
    clusterer = DBSCAN(eps=eps, metric="euclidean", n_jobs=jobs)
    clusterer.fit(encodings)
    return clusterer.labels_

def create_output_directory(base_path):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)
    return base_path

def extract_faces(data, labels, output_dir="ClusteredFaces", montage_dir="Montage"):
    output_path = create_output_directory(output_dir)
    montage_path = os.path.join(output_path, montage_dir)
    os.makedirs(montage_path)
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        print(f"Processing faces for label: {label}")
        label_folder = os.path.join(output_path, f"Face_{label}")
        os.makedirs(label_folder)
        
        indices = np.where(labels == label)[0]
        selected_indices = np.random.choice(indices, min(25, len(indices)), replace=False)
        faces = []
        
        for i, index in enumerate(selected_indices, start=1):
            image = cv2.imread(data[index]["imagePath"])
            top, right, bottom, left = data[index]["loc"]
            face = image[top:bottom, left:right]
            face_resized = cv2.resize(face, (400, int(400 * (bottom - top) / (right - left))))
            
            face_filename = os.path.join(label_folder, f"face_{i}.jpg")
            cv2.imwrite(face_filename, face_resized)
            faces.append(face_resized)
        
        if faces:
            montage = build_montages(faces, (96, 120), (5, 5))[0]
            cv2.imwrite(os.path.join(montage_path, f"Face_{label}.jpg"), montage)

if __name__ == "__main__":
    encoding_file = "encodings.pickle"
    data = load_encodings(encoding_file)
    encodings = [entry["encoding"] for entry in data]
    labels = cluster_faces(encodings)
    extract_faces(data, labels)
