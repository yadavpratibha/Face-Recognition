# Face Data Collection using OpenCV  

This project captures real-time face images using OpenCV and saves them as `.npy` files for future face recognition tasks.  

## 🚀 Features  
- Real-time face detection with Haar Cascade Classifier  
- Automatic face cropping and resizing to `100x100` pixels  
- Stores face data in `.npy` format for easy training and recognition  
- Press **'q'** to stop capturing  

## 🛠️ Requirements  
- Python 3.x  
- OpenCV (`cv2`)  
- NumPy  

Install dependencies:  
```bash
pip install opencv
pip install numpy
```  

## ▶️ Usage  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yadavpratibha/face-data-collection.git
   cd face-data-collection
   ```
2. Run the script:  
   ```bash
   python face_capture.py
   ```
3. Enter the **name of the person** when prompted.  
4. Captured face data will be saved in `./data/{name}.npy`.  

## 📸 Output Example  
- Webcam opens and shows real-time detection with a bounding box.  
- Cropped face images are displayed separately.  
- Data is saved as:  
  ```
  (num_samples, 10000)   # Flattened 100x100 face images
  ```

- Add support for multiple users in one run.  
- Improve detection with DNN-based face detectors.  
