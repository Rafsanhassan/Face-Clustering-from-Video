import os
import time
import shutil
from FaceClusteringLibrary import FramesGenerator, Pipeline, FramesProvider, FaceEncoder, DatastoreManager, TqdmUpdate, PicklesListCollator, FaceClusterUtility, FaceImageGenerator

def process_video(video_path="Footage.mp4", frames_dir="Frames", encodings_dir="Encodings", output_pickle="encodings.pickle"):
    frames_generator = FramesGenerator(video_path)
    frames_generator.GenerateFrames(frames_dir)
    
    current_path = os.getcwd()
    frames_path = os.path.join(current_path, frames_dir)
    encodings_path = os.path.join(current_path, encodings_dir)
    
    if os.path.exists(encodings_path):
        shutil.rmtree(encodings_path, ignore_errors=True)
        time.sleep(0.5)
    os.makedirs(encodings_path)
    
    pipeline = Pipeline(
        FramesProvider("Files source", sourcePath=frames_path) |
        FaceEncoder("Encode faces") |
        DatastoreManager("Store encoding", encodingsOutputPath=encodings_path),
        n_threads=3, quiet=True)
    
    pbar = TqdmUpdate()
    pipeline.run(update_callback=pbar.update)
    print("\n[INFO] Encodings extracted")
    
    if os.path.exists(output_pickle):
        os.remove(output_pickle)
    
    collator = PicklesListCollator(encodings_path)
    collator.GeneratePickle(output_pickle)
    
    time.sleep(0.5)  # Allow time for file writing
    
    face_cluster = FaceClusterUtility(output_pickle)
    face_images = FaceImageGenerator(output_pickle)
    
    labels = face_cluster.Cluster()
    face_images.GenerateImages(labels, "ClusteredFaces", "Montage")

if __name__ == "__main__":
    process_video()
