import cv2
import os
import numpy as np

def prepare_training_data(data_folder, target_size=(100, 100)):
    faces = []
    labels = []
    label_dict = {}

    for label, image_file in enumerate(os.listdir(data_folder)):
        label_dict[label] = image_file.split('.')[0]
        image_path = os.path.join(data_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Resize the image to the target size
            image = cv2.resize(image, target_size)
            faces.append(image)
            labels.append(label)

    return faces, labels, label_dict


def train_model(faces, labels, epochs=10, test_size=0.2):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    accuracy_matrix = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        faces = np.array(faces)
        labels = np.array(labels)

        # Split data into training and testing sets
        train_indices = np.random.choice(len(faces), int((1 - test_size) * len(faces)), replace=False)
        test_indices = np.array(list(set(range(len(faces))) - set(train_indices)))

        train_faces = faces[train_indices]
        train_labels = labels[train_indices]
        test_faces = faces[test_indices]
        test_labels = labels[test_indices]

        # Train the recognizer
        recognizer.train(train_faces, train_labels)

    recognizer.save("trainer.yml")
    print("Model trained and saved successfully.")
    return accuracy_matrix

data_folder = "C:/Users/.../OneDrive/Desktop/../data"  # Change this to the path of your data folder
faces, labels, label_dict = prepare_training_data(data_folder)

epochs = 5  # Define the number of epochs
accuracy_matrix = train_model(faces, labels, epochs=epochs)

print("Label dictionary:", label_dict)

