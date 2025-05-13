import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class PhysicsInformedUNet:
    def __init__(self, input_shape, num_filters=64, num_classes=1):
        """
        Initialize Physics-Informed U-Net model
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_filters (int): Number of filters in the first layer
            num_classes (int): Number of output classes/channels
        """
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _conv_block(self, inputs, filters, kernel_size=3, padding='same'):
        """Convolutional block with batch normalization and ReLU activation"""
        x = layers.Conv2D(filters, kernel_size, padding=padding)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def _encoder_block(self, inputs, filters, kernel_size=3):
        """Encoder block with skip connection"""
        x = self._conv_block(inputs, filters, kernel_size)
        x = self._conv_block(x, filters, kernel_size)
        skip = x
        x = layers.MaxPooling2D((2, 2))(x)
        return x, skip
    
    def _decoder_block(self, inputs, skip_features, filters, kernel_size=3):
        """Decoder block with skip connection"""
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
        x = layers.Concatenate()([x, skip_features])
        x = self._conv_block(x, filters, kernel_size)
        x = self._conv_block(x, filters, kernel_size)
        return x
    
    def _build_model(self):
        """Build the U-Net architecture"""
        inputs = layers.Input(self.input_shape)
        
        # Encoder
        x1, skip1 = self._encoder_block(inputs, self.num_filters)
        x2, skip2 = self._encoder_block(x1, self.num_filters * 2)
        x3, skip3 = self._encoder_block(x2, self.num_filters * 4)
        x4, skip4 = self._encoder_block(x3, self.num_filters * 8)
        
        # Bridge
        x5 = self._conv_block(x4, self.num_filters * 16)
        
        # Decoder
        x6 = self._decoder_block(x5, skip4, self.num_filters * 8)
        x7 = self._decoder_block(x6, skip3, self.num_filters * 4)
        x8 = self._decoder_block(x7, skip2, self.num_filters * 2)
        x9 = self._decoder_block(x8, skip1, self.num_filters)
        
        # Output
        outputs = layers.Conv2D(self.num_classes, (1, 1), padding='same', activation='sigmoid')(x9)
        
        return Model(inputs=inputs, outputs=outputs, name='physics_informed_unet')
    
    def physics_loss(self, y_true, y_pred, physics_weight=0.1):
        """
        Custom loss function that combines MSE with physics-based constraints
        
        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
            physics_weight: Weight for the physics-based loss term
            
        Returns:
            Combined loss value
        """
        # Standard MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # Physics-based loss (example: gradient consistency)
        # This is a simple example - modify based on your specific physics constraints
        grad_true = tf.image.image_gradients(y_true)
        grad_pred = tf.image.image_gradients(y_pred)
        physics_loss = tf.reduce_mean(tf.square(grad_true - grad_pred))
        
        # Combine losses
        total_loss = mse_loss + physics_weight * physics_loss
        return total_loss
    
    def compile_model(self, learning_rate=1e-4):
        """Compile the model with custom loss function"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=self.physics_loss,
            metrics=['mse', 'mae']
        )
        return self.model
    
    def train(self, train_dataset, val_dataset, epochs, batch_size, callbacks=None):
        """Train the model with physics-informed loss"""
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )
            ]
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return history 