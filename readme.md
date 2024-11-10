# Campus Vision Challenge - Image Classification

## Team Details
- **Team Name**: team1
- **Members**:
  - Safal Dhamala
  - Lalit Yadav

## Project Overview
This project focuses on building an image classification model to identify and classify 10 buildings on the Mississippi State University campus. The goal is to train a convolutional neural network (CNN) using the ResNet-18 architecture, which is a popular choice for image recognition tasks due to its balanced performance in both accuracy and training efficiency.

## Folder Structure
The project folder structure should look like this:
```bash
CampusVisionChallenge/
├── models/                                   # Folder containing the trained models
│   └── resnet-pretrained-large.pth           # The best model saved after training
├── log.txt                                   # Log file for training process
├── requirements.txt                          # List of dependencies (e.g., torch, torchvision, etc.)
├── README.md                                 # Project documentation
├── resnet_pretrained.ipynb                   # Jupyter notebook for working with ResNet
├── ipynb_checkpoints/                        # Folder for Jupyter notebook checkpoint files (optional)
├── __pycache__/                              # Python bytecode cache (optional, can be ignored)
└──.gitignore                                 # Git ignore file to prevent tracking unnecessary files
```


### Approach
1. **Data Preprocessing**:
   - The dataset was organized into folders based on building names, with each folder representing a class.
   - Data augmentation techniques such as random cropping, horizontal flipping, and normalization were applied to enhance generalization and prevent overfitting.

2. **Model Architecture**:
   - We used the pre-trained **ResNet-18** model, replacing its final layer to match the number of classes in our dataset.
   - This model was fine-tuned on our dataset for optimal results.

3. **Training Process**:
   - The model was trained using Cross-Entropy Loss and optimized with SDG optimizer.
   - The model was evaluated across 15 epochs, with metrics calculated for training and validation phases.
   - We trained some models with limited amount of data to tune the hyperparameters.
   - **The final model took nearly 8 hours to train on a CPU with 16 cores and 128 GB of RAM.**

### Model Performance
After training, the model achieved:
- **Validation Accuracy**: 98.49%
- **Metrics**:
  - **Precision**: 0.9850
  - **Recall**: 0.9849
  - **F1 Score**: 0.9849
  - **Log Loss**: 0.0545
## Requirements
To run the project, install the dependencies listed in `requirements.txt` and run the Jupyter Notebook file "**resnet-pretrained.ipynb**":
```bash
pip install -r requirements.txt
```

## Notes on Testing the Model
The model returns integers from 0-9 each corresponding to one of the 10 building classes. The buildings are numbered in order of their arrangement in the directory of images. For reference:
```bash
0: Butler
1: Carpenter
2: Lee
3: McCain
4: McCool
5: Old Main
6: Simrall
7: Student Union
8: Swalm
9: Walker
```
