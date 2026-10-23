"""
Handwritten Digit Recognition - Model Training Script.
Phase 2: MNIST dataset loading, inspection, preprocessing, CNN architecture definition,
training, evaluation, model export, and reload verification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_SEED = 42
MODEL_FILE_PATH = "digit_model.keras"


def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds across libraries for reproducible training."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def load_and_preprocess_data():
    """
    Load MNIST dataset, verify labels (0-9), normalize pixel values to [0, 1],
    and reshape to (N, 28, 28, 1) for the CNN input layer.
    """
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print(f"Original x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
    print(f"Original x_test shape:  {x_test.shape}, dtype: {x_test.dtype}")

    # Verify labels 0-9
    unique_train_labels = np.unique(y_train)
    unique_test_labels = np.unique(y_test)
    expected_labels = np.arange(10)
    print(f"Training unique labels: {unique_train_labels.tolist()}")
    print(f"Test unique labels:     {unique_test_labels.tolist()}")

    assert np.array_equal(unique_train_labels, expected_labels), "Train labels do not cover 0-9 exactly!"
    assert np.array_equal(unique_test_labels, expected_labels), "Test labels do not cover 0-9 exactly!"
    print("Label verification passed: labels exactly match digits 0 through 9.")

    # Normalize pixel values to 0.0 - 1.0
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    print(f"Normalized pixel range: [{x_train.min():.1f}, {x_train.max():.1f}]")

    # Reshape images to (height, width, channels) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    print(f"Processed x_train tensor shape: {x_train.shape}")
    print(f"Processed x_test tensor shape:  {x_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10) -> keras.Model:
    """
    Build a Convolutional Neural Network (CNN) with Conv2D, MaxPooling2D,
    Dropout, Flatten, and Dense layers.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape, name="digit_input"),
            # First Conv block: Feature extraction (edges, basic curves)
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv_1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool_1"),
            # Second Conv block: High-level digit features (loops, intersections)
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool_2"),
            layers.Dropout(0.25, name="dropout_conv"),
            # Classifier head
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense_hidden"),
            layers.Dropout(0.5, name="dropout_dense"),
            layers.Dense(num_classes, activation="softmax", name="digit_probabilities"),
        ],
        name="Handwritten_Digit_CNN",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(
    epochs: int = 6,
    batch_size: int = 128,
    save_path: str = MODEL_FILE_PATH
):
    """Orchestrate reproducible data loading, model compilation, training, and evaluation."""
    set_seed(RANDOM_SEED)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    print("\n--- Model Architecture ---")
    model = build_cnn_model()
    model.summary()

    print(f"\n--- Training CNN for {epochs} epochs (batch_size={batch_size}) ---")
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1,
    )

    print("\n--- Evaluating on Held-Out Test Set (10,000 samples) ---")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Actual Test Loss:     {test_loss:.4f}")
    print(f"Actual Test Accuracy: {test_accuracy * 100:.2f}%")

    print(f"\n--- Saving Trained Model to {save_path} ---")
    model.save(save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved model successfully. File size: {file_size_mb:.2f} MB")

    print("\n--- Verifying Model Reload and Inference ---")
    reloaded_model = keras.models.load_model(save_path)
    sample_index = 0
    sample_input = x_test[sample_index : sample_index + 1]
    true_label = y_test[sample_index]

    probabilities = reloaded_model.predict(sample_input, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))

    print(f"Sample index: {sample_index}")
    print("Predicted class probabilities (10 classes):")
    for digit, prob in enumerate(probabilities):
        print(f"  Digit {digit}: {prob:.4f}")
    print(f"Predicted digit: {predicted_digit} (Confidence: {probabilities[predicted_digit]*100:.2f}%)")
    print(f"True test label: {true_label}")

    assert len(probabilities) == 10, "Output must contain exactly 10 class probabilities"
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-3), "Probabilities must sum to 1.0"
    print("Reloaded model verification passed: outputs 10 valid probabilities.")

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "history": history.history,
    }


def main():
    """Main execution function."""
    train_and_evaluate()


if __name__ == "__main__":
    main()
