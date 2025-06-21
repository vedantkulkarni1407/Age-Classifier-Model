import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tqdm import tqdm

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "assessment_data"
AUGMENT_FACTOR = 5

def load_dataset(dataset_path):
    images, labels = [], []
    print("Loading and labeling images...")
    for age_label in sorted(os.listdir(dataset_path), key=lambda x: int(x)):
        folder_path = os.path.join(dataset_path, age_label)
        if not os.path.isdir(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, filename)
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
                img = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img)
                labels.append(age_label)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
    return np.array(images), np.array(labels)

def augment_data(images, labels, factor=AUGMENT_FACTOR):
    print("Augmenting images...")
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                 zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
    augmented_images, augmented_labels = [], []
    for i in tqdm(range(len(images))):
        img = images[i].reshape((1, *images[i].shape))
        label = labels[i]
        count = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            count += 1
            if count >= factor:
                break
    return np.array(augmented_images), np.array(augmented_labels)

def extract_features(images):
    print("Extracting features using MobileNetV2...")
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    images = preprocess_input(images)
    features = model.predict(images, batch_size=BATCH_SIZE, verbose=1)
    return features

def main():
    # Step 1: Load data
    images, labels = load_dataset(DATASET_PATH)

    # Step 2: Augment
    augmented_images, augmented_labels = augment_data(images, labels)

    # Combine original + augmented
    all_images = np.concatenate((images, augmented_images), axis=0)
    all_labels = np.concatenate((labels, augmented_labels), axis=0)

    # Step 3: Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(all_labels)

    # Step 4: Split
    X_train, X_temp, y_train, y_temp = train_test_split(all_images, encoded_labels, test_size=0.3, stratify=encoded_labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Step 5: Feature extraction
    X_train_feat = extract_features(X_train)
    X_val_feat = extract_features(X_val)
    X_test_feat = extract_features(X_test)

    # Step 6: Save
    np.save("X_train.npy", X_train_feat)
    np.save("y_train.npy", y_train)
    np.save("X_val.npy", X_val_feat)
    np.save("y_val.npy", y_val)
    np.save("X_test.npy", X_test_feat)
    np.save("y_test.npy", y_test)
    np.save("label_encoder.npy", le.classes_)

    print("✅ Data preprocessing complete. Features and labels saved.")

if __name__ == "__main__":
    main()
