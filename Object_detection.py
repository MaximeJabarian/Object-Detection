import sys
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt


print(sys.version)


def unpickle(file):
    """
    Unpickle a file and return the dictionary inside.

    Args:
        file (str): The path to the file.

    Returns:
        dict: The dictionary from the file.
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def process_files(file_list):
    """
    Process a list of files and return a combined dictionary.

    Args:
        file_list (list): The list of file paths.

    Returns:
        dict: The combined dictionary.
    """
    combined_dict = {'labels': [], 'data': [], 'filenames': []}

    for file in file_list:
        data_dict = unpickle(file)
        data_dict[b'data'] = [inner_arr for inner_arr in data_dict[b'data']]
        combined_dict['labels'].extend(data_dict[b'labels'])
        combined_dict['data'].extend(data_dict[b'data'])
        combined_dict['filenames'].extend(data_dict[b'filenames'])

    return combined_dict


def preprocess_data(data):
    """
    Preprocess images and labels.

    Args:
        data (DataFrame): The DataFrame containing the data.

    Returns:
        tuple: A tuple containing the preprocessed images and labels.
    """
    images = []
    labels = []

    for _, row in data.iterrows():
        image = np.array(row['data'], dtype=np.uint8).reshape((32, 32, 3))
        label = row['labels']
        images.append(image)
        labels.append(label)

    images = np.array(images) / 255.0
    labels = np.array(labels)

    return images, labels


def create_model():
    """
    Create a TensorFlow model for object detection.

    Returns:
        tf.keras.Sequential: The TensorFlow model.
    """
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_confusion_matrix(model, df_test, class_names):
    """
    Plot the confusion matrix for a TensorFlow model.

    Args:
        model (tf.keras.Sequential): The TensorFlow model.
        df_test (DataFrame): The DataFrame containing the test data.
        class_names (list): The list of class names.
    """
    images, labels = preprocess_data(df_test)
    X_test, X_, y_test, y_ = train_test_split(images, labels, test_size=1, random_state=42)

    # Test the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    # Get the predicted probabilities for each class
    y_pred_probs = model.predict(X_test)

    # Get the predicted labels using argmax
    y_pred = np.argmax(y_pred_probs

    # Get the predicted labels using argmax
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Create a DataFrame with the correct and predicted labels
    comparison_df = pd.DataFrame({'Correct': y_test, 'Predicted': y_pred})

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Normalize the confusion matrix to get percentages instead of raw counts
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a heatmap to visualize the confusion matrix
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Correct')
    plt.title('Confusion Matrix')
    plt.show()

# Load and process the data
files = ["batches.meta", "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
file_list = files[1:6]  
combined_data = process_files(file_list)
df = pd.DataFrame(combined_data)

# Preprocess the data
images, labels = preprocess_data(df)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=15, validation_split=0.2)

# Test the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Load the test data
df_test_dict = unpickle("test_batch")
new_dict = {'labels': df_test_dict[b'labels'], 'data': df_test_dict[b'data'], 'filenames': df_test_dict[b'filenames']}
df_test = pd.DataFrame(new_dict)

# Plot the confusion matrix
class_names = unpickle(files[0])
class_names = pd.DataFrame(class_names)[b'label_names']
plot_confusion_matrix(model, df_test, class_names)
