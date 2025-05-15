import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# CSV Logging setup
RESULTS_FILE = 'model_results.csv'
CSV_HEADERS = [
    'epochs', 'batch_size', 'conv_layers', 'fc_layers', 'optimizer',
    'test_accuracy', 'test_loss', 'precision', 'recall'
]

if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADERS)

# Load datasets
def load_datasets(image_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'processed_dataset/train',
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        'processed_dataset/val',
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        'processed_dataset/test',
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    return train_ds, val_ds, test_ds

# Build model dynamically
def build_model(conv_layers, fc_layers, optimizer_name, input_shape=(512, 512, 3)):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1. / 255))

    filters = 32
    for _ in range(conv_layers):
        model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        filters *= 2

    model.add(layers.Flatten())

    for _ in range(fc_layers):
        model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    if optimizer_name == 'adam':
        optimizer = optimizers.Adam()
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD()
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Plot training and validation accuracy/loss
def plot_metrics(history, title_suffix=""):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation Accuracy {title_suffix}')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Training and Validation Loss {title_suffix}')

    plt.show()

# Evaluate model and log to CSV
def evaluate_and_log(model, test_ds, params):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend((preds > 0.5).astype(int).flatten())

    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    print(f"Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Log results to CSV
    with open(RESULTS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            params['epochs'],
            params['batch_size'],
            params['conv_layers'],
            params['fc_layers'],
            params['optimizer'],
            f"{test_acc:.4f}",
            f"{test_loss:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}"
        ])

# Main loop to run experiments
def run_experiments():
    image_size = (512, 512)
    input_shape = (512, 512, 3)

    epochs_list = [3, 6, 9, 12]
    batch_sizes = [64, 128, 256]
    conv_layers_list = [2, 3]
    fc_layers_list = [1, 2]
    optimizer = 'adam'

    for batch_size in batch_sizes:
        print(f"\n=== Loading dataset with batch size: {batch_size} ===")
        train_ds, val_ds, test_ds = load_datasets(image_size, batch_size)

        for epochs in epochs_list:
            for conv_layers in conv_layers_list:
                for fc_layers in fc_layers_list:
                    print(f"\n=== Training model: Epochs={epochs}, Batch={batch_size}, Conv={conv_layers}, FC={fc_layers}, Optimizer={optimizer} ===")
                    model = build_model(conv_layers, fc_layers, optimizer, input_shape)

                    history = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        verbose=0  # Change to 1 for detailed training logs
                    )

                    title = f"(E{epochs}-B{batch_size}-C{conv_layers}-F{fc_layers}-{optimizer})"
                    plot_metrics(history, title_suffix=title)

                    params = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'conv_layers': conv_layers,
                        'fc_layers': fc_layers,
                        'optimizer': optimizer
                    }

                    evaluate_and_log(model, test_ds, params)


# Execute Code
run_experiments()
