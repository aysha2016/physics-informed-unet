import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # Encoding path (downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    # Decoding path (upsampling)
    u1 = UpSampling2D((2, 2))(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)

    # Skip connection
    concat = Concatenate()([c1, c2])

    # Final output layer
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(concat)

    model = Model(inputs, outputs)
    return model

def train():
    model = build_unet()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.summary()

    # Dummy data for demonstration purposes
    x_train = tf.random.normal((5, 256, 256, 3))
    y_train = tf.random.normal((5, 256, 256, 3))

    model.fit(x_train, y_train, epochs=5, batch_size=1)

if __name__ == "__main__":
    train()
    