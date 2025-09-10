# Pose-to-Style: Real-Time Human Pose Driven Style Transfer

This project detects **human pose keypoints** from camera input, recognizes the current pose by comparing it to predefined actions, and applies a corresponding **style transfer** effect on the video stream.

---

## 1. Features
- Detect 18 body keypoints using **trt_pose** (ResNet18 backbone, pretrained by NVIDIA)  
- Normalize keypoint coordinates for robust pose recognition  
- Match user poses with standard poses via **MSE loss** (with masking to focus on important joints)  
- Apply **FastStyleNet** for real-time style transfer  
- Support **7 artistic styles**: scream, prismas, fur, mermaid, wukon, pop, sketch 

---

## 2. Project Architecture
Camera → Pose Detection (trt_pose) → Keypoint Normalization → Pose Recognition (MSE loss) → Pose-Triggered Style Selection → FastStyleNet → Styled Output

---

## 3. 
