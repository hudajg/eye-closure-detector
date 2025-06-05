# Sleep Detection by using python Libary 

## Project Overview: 
This project detects drowsiness using a webcam by tracking eye landmarks and calculating the Eye Aspect Ratio (EAR), with the help of Python libraries such as OpenCV for video processing, dlib for facial landmark detection, scipy for distance calculations, and pygame for audio alerts. If the eyes remain closed for a set number of frames, a sound alert is triggered. 

## 🚀 How to Run This Project

1. **Clone the repository:**

 ```bash
git clone https://github.com/hudajg/eye-closure-detector.git
cd eye-closure-detector
  ```
2. **Install required libraries:**
  ```bash
pip install opencv-python dlib pygame scipy imutils
  ```
On macOS, you may also need:
  ```bash
brew install cmake
xcode-select --install
  ```
3. **Run the program:**
  ```bash
python3 eye-closure-detector.py
  ```
4. **Exit the program:**
press ESC on Keyboard
