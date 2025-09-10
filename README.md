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

## 3. Models
### (1) Pose Detection
- Model: **trt_pose** (NVIDIA official pretrained model)  
- Backbone: **ResNet18** 
- Detects 18 human body keypoints  
- Normalization ensures poses are comparable regardless of position in frame  
- Pose recognition via MSE loss against predefined standard poses
- Loss focuses more on **hands/legs** using masking

---

### (2) Style Transfer
- Model: **FastStyleNet**  
- Dataset: MS COCO + chosen style image  
- Loss functions: 
  - **Content Loss**: Difference between transformed image & original input  
  - **Style Loss**: Difference between transformed image & style image (via Gram matrix)
- Training settings:
  - Epochs = 2
  - Batch size = 1
  - Image size = 256
- Styles trained: scream, prismas, fur, mermaid, wukon, pop, sketch
- Results:  


---

## 4. Deployment on Xavier
- Models deployed: **trt_pose + FastStyleNet**  
- Issue: Real-time performance was slow due to large image size  
- Solution: Resize input image to 50% to speed up inference  

---

## 5. Demo

