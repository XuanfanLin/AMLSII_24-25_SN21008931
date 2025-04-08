import os
import random
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# -------------------- Configuration --------------------
IMG_SIZE = 512  
SIZE = (IMG_SIZE, IMG_SIZE)
NUM_CLASSES = 5
BATCH_SIZE = 16  
# Epochs are defined for the training phase but won’t be used here
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 15

# -------------------- Utility Functions --------------------
def seed_everything(seed=21):
    """Sets seeds and deterministic options for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# -------------------- Pipeline Function --------------------
def run_pipeline():
    """
    Executes the entire processing pipeline:
    - Setup and configuration
    - Data loading and visualization
    - Generator creation and data augmentation
    - Loading of a pretrained model and its evaluation
    """
    # Setup
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    seed_everything()
    warnings.filterwarnings('ignore')

    # Define local paths (update these paths if needed)
    work_dir = os.path.join( 'Datasets', 'cassava-leaf-disease-classification')
    train_path = os.path.join(work_dir, 'train_images')
    TEST_DIR = os.path.join(work_dir, 'test_images')
    output_dir = os.path.join( 'Results')
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- Load Dataset --------------------
    data_csv = os.path.join(work_dir, 'train.csv')
    data = pd.read_csv(data_csv)
    label_map_path = os.path.join(work_dir, 'label_num_to_disease_map.json')
    with open(label_map_path) as f:
        real_labels = json.load(f)
        # Convert keys to integers
        real_labels = {int(k): v for k, v in real_labels.items()}
    data['class_name'] = data['label'].map(real_labels)
    
    # Split the data (10% for evaluation)
    train_df, test_df = train_test_split(data, test_size=0.1, random_state=42, stratify=data['class_name'])

    # -------------------- Visualize Class Distribution --------------------
    short_label_map = {0: 'CBB', 1: 'CBSD', 2: 'CGM', 3: 'CMD', 4: 'Healthy'}
    data['short_label'] = data['label'].map(short_label_map)
    
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='short_label', data=data, palette='viridis', edgecolor='black')
    plt.title("Class Distribution in Original Dataset")
    plt.xlabel("Class Label")
    plt.ylabel("Sample Count")
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f'{height}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=11, xytext=(0, 6),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

    # -------------------- Sample Images per Class --------------------
    import matplotlib.image as mpimg
    sample_dir = train_path
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    classes = sorted(data['label'].unique())
    for i, label in enumerate(classes):
        img_name = data[data['label'] == label].iloc[0]['image_id']
        img_path = os.path.join(sample_dir, img_name)
        img = mpimg.imread(img_path)
        row, col = divmod(i, 3)
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"{short_label_map[label]}", fontsize=14)
        axes[row][col].axis('off')
    for j in range(len(classes), 6):
        row, col = divmod(j, 3)
        axes[row][col].axis('off')
    plt.suptitle("Example Image from Each Class", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_samples.png"))
    plt.close()

    # -------------------- Data Generators --------------------
    # Data augmentation for training (kept for demonstration)
    datagen_train = ImageDataGenerator(
        validation_split=0.2,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    train_generator = datagen_train.flow_from_dataframe(
        train_df,
        directory=train_path,
        x_col='image_id',
        y_col='class_name',
        subset='training',
        target_size=SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42,
        batch_size=BATCH_SIZE
    )
    validation_datagen = ImageDataGenerator(
        validation_split=0.2,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    validation_generator = validation_datagen.flow_from_dataframe(
        train_df,
        directory=train_path,
        x_col='image_id',
        y_col='class_name',
        subset='validation',
        target_size=SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42,
        batch_size=BATCH_SIZE
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        directory=train_path,
        x_col='image_id',
        y_col='class_name',
        target_size=SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42,
        batch_size=BATCH_SIZE
    )
    print("\nFound {} validated image filenames belonging to {} classes.".format(
        test_generator.n, len(test_generator.class_indices)))
    
    # -------------------- Data Augmentation Visualization --------------------
    sample_aug = train_df.iloc[[0]].copy()
    sample_aug['class_name'] = sample_aug['label'].map(real_labels)
    preview_gen = datagen_train.flow_from_dataframe(
        sample_aug,
        directory=train_path,
        x_col='image_id',
        y_col='class_name',
        target_size=SIZE,
        class_mode='categorical',
        batch_size=1
    )
    aug_imgs = [preview_gen[0][0][0] / 255.0 for _ in range(4)]
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i in range(4):
        axs.flat[i].imshow(aug_imgs[i])
        axs.flat[i].axis('off')
    plt.suptitle("Data Augmentation Examples", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmentation_samples.png"))
    plt.close()
    print("\nFound {} validated image filenames in augmentation preview.".format(preview_gen.n))
    
    # -------------------- Compute Class Weights --------------------
    class_counts = train_df['label'].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = {i: total_samples / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)}
    print("Class Weights:", class_weights)
    
    # -------------------- Load Pre-trained Model --------------------
    # Instead of training, load the pre-trained model for evaluation/demonstration
    model_path = os.path.join(output_dir, 'Cassava_model_finetuned.keras')
    model = load_model(model_path)
    print("\nPre-trained model loaded successfully.")
    
    # -------------------- Model Architecture Visualization --------------------
    plot_model(model, to_file=os.path.join(output_dir, "model_architecture.png"),
               show_shapes=True, show_layer_names=True)
    
    # -------------------- Evaluation --------------------
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f"\nFinal model evaluation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    y_true = test_df['label'].values
    pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(pred_probs, axis=1)
    
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    print("\nClassification Report:")
    target_names = [real_labels[i] for i in range(NUM_CLASSES)]
    print(classification_report(y_true, y_pred, target_names=target_names))


# If run as a script, execute the pipeline.
if __name__ == '__main__':
    run_pipeline()
