import tensorflow as tf
import numpy as np
import yaml
import os
from datetime import datetime
from model import PhysicsInformedUNet
import matplotlib.pyplot as plt

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_data_generators(config):
    """Create training and validation data generators"""
    # Example data loading - modify according to your dataset
    def load_and_preprocess_data(data_path):
        data = np.load(data_path)
        # Normalize data
        data = (data - data.min()) / (data.max() - data.min())
        return data
    
    # Load your data here
    # This is a placeholder - replace with your actual data loading logic
    train_data = load_and_preprocess_data(config['data']['train_path'])
    val_data = load_and_preprocess_data(config['data']['val_path'])
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
    
    # Apply batching and prefetching
    train_dataset = train_dataset.batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

def create_callbacks(config):
    """Create training callbacks"""
    # Create directories for saving models and logs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config['training']['log_dir'], timestamp)
    model_dir = os.path.join(config['training']['model_dir'], timestamp)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'model_{epoch:02d}.h5'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['training']['lr_patience'],
            min_lr=1e-6
        )
    ]
    return callbacks

def plot_training_history(history, save_path):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Model MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Load configuration
    config = load_config()
    
    # Create data generators
    train_dataset, val_dataset = create_data_generators(config)
    
    # Initialize model
    model = PhysicsInformedUNet(
        input_shape=config['model']['input_shape'],
        num_filters=config['model']['num_filters'],
        num_classes=config['model']['num_classes']
    )
    
    # Compile model
    model.compile_model(learning_rate=config['training']['learning_rate'])
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Train model
    history = model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks
    )
    
    # Plot and save training history
    plot_training_history(
        history,
        os.path.join(config['training']['log_dir'], 'training_history.png')
    )

if __name__ == '__main__':
    main() 