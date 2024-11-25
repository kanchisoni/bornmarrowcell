import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout , Activation
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

dataset = 'D:\bone_marrow\bone_marrow_cell_dataset'

datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize images
    validation_split=0.2,  # 20% validation split
)


train_generator = datagen.flow_from_directory(
    dataset,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)
custom_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  # Adjust based on number of classes
])

custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
custom_model.summary()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

custom_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  # Adjust based on number of classes
])

custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
custom_model.summary()

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)

pretrained_model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

pretrained_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
pretrained_model.summary()

# Train custom CNN
history_custom = custom_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
)

# Train pre-trained model
history_pretrained = pretrained_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
)


test_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',  # Use validation as test if no separate test set
    shuffle=False
)


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Predict with Custom CNN
y_pred_custom = np.argmax(custom_model.predict(test_generator), axis=1)
y_true = test_generator.classes

# Predict with Pre-trained Model
y_pred_pretrained = np.argmax(pretrained_model.predict(test_generator), axis=1)

# Confusion Matrix
cm_custom = confusion_matrix(y_true, y_pred_custom)
cm_pretrained = confusion_matrix(y_true, y_pred_pretrained)

sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues')
plt.title("Custom CNN Confusion Matrix")
plt.show()

sns.heatmap(cm_pretrained, annot=True, fmt='d', cmap='Blues')
plt.title("Pre-trained Model Confusion Matrix")
plt.show()

# Classification Reports
print("Custom CNN Report:\n", classification_report(y_true, y_pred_custom))
print("Pre-trained Model Report:\n", classification_report(y_true, y_pred_pretrained))

# Plot ROC Curve
fpr_custom, tpr_custom, _ = roc_curve(y_true, custom_model.predict(test_generator).ravel())
fpr_pretrained, tpr_pretrained, _ = roc_curve(y_true, pretrained_model.predict(test_generator).ravel())

plt.plot(fpr_custom, tpr_custom, label="Custom CNN")
plt.plot(fpr_pretrained, tpr_pretrained, label="Pre-trained Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

 
 
 

  






# csv_path = 'abbreviations.csv'
# image_folder = 'bone_marrow_cell_dataset'
# df = pd.read_csv(csv_path)

# # Add full path to image files
# df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_folder, x))

# # Map labels to integers
# label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
# df['label_int'] = df['label'].map(label_map)

# # Split dataset into train, validation, and test sets
# train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_int'], random_state=42)
# train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label_int'], random_state=42)

# # Prepare TensorFlow Dataset
# def load_image(image_path, label):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, [224, 224])
#     img = img / 255.0  # Normalize to [0, 1]
#     return img, label
