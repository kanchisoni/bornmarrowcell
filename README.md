# bornmarrowcell
CNN classification
Bone Marrow Cell Classification Using CNNs
Project Overview
This project focuses on classifying bone marrow cell images into their respective categories using Convolutional Neural Networks (CNNs). The dataset consists of labeled folders, each representing a specific cell type. Two models are used for performance analysis:

A Custom CNN model built from scratch.
A Pre-trained CNN model (ResNet50) fine-tuned for this classification task.
Goal
Compare the performance of the custom-built CNN with a pre-trained CNN using evaluation metrics like:
Accuracy
Precision
Recall
F1-Score
AUC-ROC
Confusion Matrix
Dataset
The dataset is organized into folders where each folder corresponds to a specific cell type:

Copy code
dataset/
├── Class_1/
│   ├── img1.jpg
│   ├── img2.jpg
├── Class_2/
│   ├── img3.jpg
│   ├── img4.jpg
Images were resized to 224x224 for input into the CNNs.
Data was split into training, validation, and test sets.
Methods Used
1. Data Preprocessing
Used ImageDataGenerator for:
Image normalization (scaling pixel values between 0 and 1).
Data augmentation (rotation, flipping, etc.).
Split data into:
Training Set (80%)
Validation Set (20%)
Test Set (same as validation if no separate test data was provided).
2. Models
Custom CNN
A Sequential CNN model with layers for convolution, max-pooling, dropout, and dense layers for classification:
python
Copy code
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
Pre-trained CNN (ResNet50)
ResNet50, a pre-trained model, was fine-tuned by freezing its initial layers and adding a custom classification head:
python
Copy code
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False
3. Evaluation Metrics
Confusion Matrix: Shows true positives, false positives, true negatives, and false negatives.
Classification Report: Includes precision, recall, F1-score, and support for each class.
AUC-ROC Curve: Evaluates the model's ability to distinguish between classes.
Accuracy, Precision, Recall, F1-Score: Metrics calculated on the test set.
4. Results
Metrics for both models were calculated and tabulated for comparison:

Metric	Custom CNN	Pre-trained CNN
Accuracy	...	...
Precision	...	...
Recall	...	...
F1-Score	...	...
AUC-ROC Score	...	...
Setup Instructions
Requirements
Python 3.7+
TensorFlow 2.x
Keras
NumPy
Matplotlib
Seaborn
scikit-learn
Install dependencies:

bash
Copy code
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
Run the Code
Clone the repository:
bash
Copy code
git clone https://github.com/username/bone-marrow-cell-classification.git
cd bone-marrow-cell-classification
Place the dataset in the dataset/ folder.
Run the script:
bash
Copy code
python main.py
Code Structure
bash
Copy code
bone-marrow-cell-classification/
├── dataset/                    # Dataset folder (organized into subfolders per class)
├── main.py                     # Main script to train and evaluate models
├── custom_cnn.py               # Code for the custom CNN model
├── pretrained_cnn.py           # Code for the pre-trained CNN model
├── utils.py                    # Helper functions for data loading and evaluation
├── README.md                   # Project documentation
Example Output
Confusion Matrix

AUC-ROC Curve
