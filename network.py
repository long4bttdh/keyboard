import tensorflow as tf
from tensorflow.keras import layers, models
import keras_nlp

class CoAtNet(tf.keras.Model):
    def __init__(self, num_classes=36):
        super(CoAtNet, self).__init__()

        # Convolutional part
        self.conv_layers = models.Sequential([
            layers.Conv2D(32, kernel_size=3, strides=1, padding='same', input_shape=(None, None, 1)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        ])

        # Transformer part
        self.encoder_layer = keras_nlp.layers.TransformerEncoder(num_heads=8, key_dim=32, ff_dim=32, dropout=0.1)
        self.transformer_layers = [self.encoder_layer for _ in range(2)]

        # Linear classifier
        self.fc = layers.Dense(num_classes)

    def call(self, x):
        # Convolutional layers
        x = self.conv_layers(x)

        # Reshaping for the transformer encoder
        x = tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1]))  # (batch_size, seq_length, features)

        # Transformer encoder
        for layer in self.transformer_layers:
            x = layer(x)

        # Max pooling over time
        x = tf.reduce_max(x, axis=1)

        # Classifier
        x = self.fc(x)
        return x

# Example usage