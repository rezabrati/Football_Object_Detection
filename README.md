# Football Object Detection  
This project utilizes advanced AI techniques to detect and analyze football-related objects in images and videos. The solution is designed for applications in sports analytics, video processing, and automated commentary systems.

---

## Modules Used  
The following modules and techniques are employed in this project:  
- **YOLO v8:** A state-of-the-art object detection model for identifying players, the ball, and other football-related entities.  
- **K-Means Clustering:** Used for pixel segmentation and clustering, enabling detection of t-shirt colors for team identification.  

---

## Trained Models  
- **YOLO v8 Model:** Pre-trained and fine-tuned on football datasets for high-accuracy detection.

---

## Sample Input and Output  
- **Sample Input Video:** Includes raw video footage of a football match.  
- **Sample Output Video:** Displays bounding boxes, team colors, and player movement analysis.

---

## Requirements  
To run this project, ensure the following dependencies are installed:

- **Python 3.x**  
- **ultralytics** (for YOLO model support)  
- **supervision** (for post-processing and visualization)  
- **OpenCV** (for image and video handling)  
- **NumPy** (for numerical operations)  
- **Matplotlib** (for plotting and visualization)  
- **Pandas** (for data analysis and statistics)

Install all dependencies using:

```bash
pip install -r requirements.txt
