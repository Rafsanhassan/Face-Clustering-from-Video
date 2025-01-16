# Face Clustering from Video

## Overview
**Face Clustering from Video** is a machine learning pipeline that automatically extracts faces from a video, encodes them using deep learning models, and clusters similar faces together. This is useful for applications such as security monitoring, event analysis, and face-based content organization.

## Features
- Extracts frames from a video file
- Detects and encodes faces in each frame
- Stores face encodings efficiently
- Clusters similar faces using DBSCAN
- Generates organized face clusters with annotated images
- Creates montage images of clustered faces for visualization

## Installation
### Prerequisites
Ensure you have Python installed (>=3.7) and install the required dependencies:
```bash
pip install numpy opencv-python scikit-learn dlib tqdm imutils face_recognition_models pyPiper
```

## Usage
### 1. Extract Frames from Video
The `FramesGenerator` class extracts frames from a given video file while resizing them for efficient processing.
```python
framesGenerator = FramesGenerator("Footage.mp4")
framesGenerator.GenerateFrames("Frames")
```

### 2. Face Encoding and Storage
A pipeline is designed to extract face encodings and store them efficiently.
```python
pipeline = Pipeline(
    FramesProvider("Files source", sourcePath="Frames") |
    FaceEncoder("Encode faces") |
    DatastoreManager("Store encoding", encodingsOutputPath="Encodings"),
    n_threads=3, quiet=True
)
pipeline.run()
```

### 3. Merge Encodings
After encoding, multiple pickle files are merged into a single file for clustering.
```python
picklesListCollator = PicklesListCollator("Encodings")
picklesListCollator.GeneratePickle("encodings.pickle")
```

### 4. Perform Face Clustering
The `FaceClusterUtility` clusters the detected faces using **DBSCAN**.
```python
faceClusterUtility = FaceClusterUtility("encodings.pickle")
labelIDs = faceClusterUtility.Cluster()
```

### 5. Generate Clustered Faces and Montages
Finally, the `FaceImageGenerator` organizes detected faces into labeled clusters and generates montage images.
```python
faceImageGenerator = FaceImageGenerator("encodings.pickle")
faceImageGenerator.GenerateImages(labelIDs, "ClusteredFaces", "Montage")
```

## Pipeline Workflow
1. **Frame Extraction**: Extracts frames from the video at regular intervals.
2. **Face Encoding**: Detects and extracts facial embeddings.
3. **Storage**: Saves encodings in a structured format.
4. **Merging Encodings**: Combines multiple encoding files into one.
5. **Clustering**: Uses **DBSCAN** to group similar faces.
6. **Visualization**: Generates face clusters and montage images.

## Project Structure
```
├── FaceClusteringLibrary.py  # Main face clustering library
├── process_video.py          # Main script to run the pipeline
├── Frames/                   # Extracted frames from the video
├── Encodings/                 # Face encodings stored in pickle format
├── ClusteredFaces/            # Output folder with clustered faces
├── Montage/                   # Montage images of clusters
└── README.md                 # Project documentation
```

## Dependencies
- Python (>=3.7)
- OpenCV
- dlib
- Scikit-learn
- tqdm
- imutils
- face_recognition_models

## License
This project is open-source and can be modified and distributed freely.

## Future Enhancements
- Support for real-time video processing
- More advanced clustering algorithms
- GUI-based visualization for better usability

## Contact
For questions or contributions, feel free to reach out or submit an issue on GitHub.

---
Happy coding! 🚀

