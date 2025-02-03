# Yoga Pose Detection with CNN 🧘

A real-time yoga pose detection system using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. Detects **5 yoga poses**: Downdog, Goddess, Plank, Tree, and Warrior2.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure 📁
yoga-pose-detection/
-  dataset/
-  src/
  - train_model.py
  - evaluate_model.py
  -  detect_pose.py 
- models/ 
- docs/
  - images/
- requirements.txt

## Installation ⚙️
1. **Clone the repository**:
   ```bash
   - git clone https://github.com/your_username/yoga-pose-detection.git
   - cd yoga-pose-detection
   - pip install -r requirements.txt
2. Download and extract zip file

## Usage 
- Train the Model- run "python src/train_model.py"
- Best model saved to models/best_model.keras
- Final model saved to models/yoga_pose_model.keras
- python src/evaluate_model.py
- python src/detect_pose.py
