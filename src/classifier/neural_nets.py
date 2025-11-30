import cv2
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from keras import Layer, Model, callbacks, layers, losses, models


class Conv2dMaxPoolingLayer(Layer):
    """Layer combining a convolutional layer and a max pooling layer, with optional dropout"""

    def __init__(
        self,
        filters: int,
        kernel_size: tuple[int, int] = (3, 3),
        pool_size: tuple[int, int] = (2, 2),
        use_dropout=True,
    ) -> None:
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.use_dropout = use_dropout

        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
        )
        self.max = layers.MaxPooling2D(pool_size=pool_size)

        if use_dropout:
            self.dropout = layers.Dropout(0.25)

    def build(self, input_shape):
        # Build the sub-layers
        self.conv.build(input_shape)
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.max.build(conv_output_shape)
        max_output_shape = self.max.compute_output_shape(conv_output_shape)
        if self.use_dropout:
            self.dropout.build(max_output_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.max(x)
        if self.use_dropout:
            x = self.dropout(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        # Chain the output shape computations
        shape = self.conv.compute_output_shape(input_shape)
        shape = self.max.compute_output_shape(shape)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
            }
        )
        return config


class TumourNet(Model):
    """Simple convolutional neural network for tumour classification"""

    def __init__(self, use_augmentation: bool = True):
        super().__init__()
        # Data augmentation layers (only active during training)
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.augmentation = [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.1),
                layers.RandomBrightness(0.2),
                layers.RandomContrast(0.2),
                layers.RandomGaussianBlur(),
                layers.GaussianNoise(0.05),
            ]

        self.rescaling = layers.Rescaling(1.0 / 255)
        self.conv1 = Conv2dMaxPoolingLayer(32, (3, 3), use_dropout=use_augmentation)
        self.conv2 = Conv2dMaxPoolingLayer(64, (3, 3), use_dropout=use_augmentation)
        self.conv3 = Conv2dMaxPoolingLayer(64, (2, 2), use_dropout=use_augmentation)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.dense3 = layers.Dense(4, activation="softmax")

    def call(self, x, training=None):
        x, _ = self.activations_call(x, training)
        return x

    def activations_call(self, x, training=None):
        if self.use_augmentation:
            for aug_layer in self.augmentation:
                x = aug_layer(x, training=training)

        x = self.rescaling(x)
        conv_1_act = self.conv1(x, training=training)
        conv_2_act = self.conv2(conv_1_act, training=training)
        conv_3_act = self.conv3(conv_2_act, training=training)
        x = self.flatten(conv_3_act)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        activations = (conv_1_act, conv_2_act, conv_3_act)
        return (x, activations)


class SqueezeExciteBlock(Layer):
    """Squeeze excite block implementation"""

    def __init__(
        self,
        filters: int,
        ratio: int = 16,
    ) -> None:
        super().__init__()
        self.filters = filters
        self.ratio = ratio

        self.pooling = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(filters // ratio, activation="relu")
        self.dense2 = layers.Dense(filters, activation="sigmoid")
        self.reshape = layers.Reshape((1, 1, filters))
        self.multiply = layers.Multiply()

    def call(self, x):
        se = self.pooling(x)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        x = self.multiply([x, se])
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "ratio": self.ratio,
            }
        )
        return config


class TumourNetSE(Model):
    """More complex convolutional network with squeeze excite blocks for tumour classification"""

    def __init__(self, use_augmentation: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.augmentation = [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.1),
                layers.RandomBrightness(0.2),
                layers.RandomContrast(0.2),
                layers.RandomGaussianBlur(),
                layers.GaussianNoise(0.05),
            ]

        self.rescaling = layers.Rescaling(1.0 / 255)

        # Block 1
        self.conv1 = layers.Conv2D(16, (3, 3), padding="same", activation="relu")
        self.conv2 = layers.Conv2D(16, (3, 3), padding="same")
        self.bn2 = layers.BatchNormalization()
        self.se1 = SqueezeExciteBlock(16)

        # Block 2
        self.residual2 = layers.Conv2D(32, (1, 1), strides=(2, 2), padding="same")
        self.conv3 = layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="same", activation="relu"
        )
        self.conv4 = layers.Conv2D(32, (3, 3), padding="same")
        self.bn4 = layers.BatchNormalization()
        self.se2 = SqueezeExciteBlock(32)

        # Block 3
        self.residual3 = layers.Conv2D(64, (1, 1), strides=(2, 2), padding="same")
        self.conv5 = layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding="same", activation="relu"
        )
        self.conv6 = layers.Conv2D(64, (3, 3), padding="same")
        self.bn6 = layers.BatchNormalization()
        self.se3 = SqueezeExciteBlock(64)

        # Block 4
        self.residual4 = layers.Conv2D(128, (1, 1), strides=(2, 2), padding="same")
        self.conv7 = layers.Conv2D(
            128, (3, 3), strides=(2, 2), padding="same", activation="relu"
        )
        self.conv8 = layers.Conv2D(128, (3, 3), padding="same")
        self.bn8 = layers.BatchNormalization()
        self.se4 = SqueezeExciteBlock(128)

        # Block 5
        self.residual5 = layers.Conv2D(256, (1, 1), strides=(2, 2), padding="same")
        self.conv9 = layers.Conv2D(
            256, (3, 3), strides=(2, 2), padding="same", activation="relu"
        )
        self.conv10 = layers.Conv2D(256, (3, 3), padding="same")
        self.bn10 = layers.BatchNormalization()
        self.se5 = SqueezeExciteBlock(256)

        self.pooling = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(128, activation="relu")
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dense3 = layers.Dense(64, activation="relu")
        self.dropout3 = layers.Dropout(dropout_rate)
        self.dense4 = layers.Dense(4, activation="softmax")

    def call(self, x, training=None):
        x, _ = self.activations_call(x, training)
        return x

    def activations_call(self, x, training=None):
        if self.use_augmentation:
            for aug_layer in self.augmentation:
                x = aug_layer(x, training=training)

        x = self.rescaling(x)

        # Block 1 (no residual - same dimensions)
        res1 = self.conv1(x, training=training)
        x = self.conv2(res1, training=training)
        x = self.bn2(x, training=training)
        x = layers.ReLU()(x)
        x = self.se1(x)
        block_1_act = layers.Add()([x, res1])

        # Block 2
        res2 = self.residual2(block_1_act, training=training)
        x = self.conv3(block_1_act, training=training)
        x = self.conv4(x, training=training)
        x = self.bn4(x, training=training)
        x = layers.ReLU()(x)
        x = self.se2(x)
        block_2_act = layers.Add()([x, res2])

        # Block 3
        res3 = self.residual3(block_2_act, training=training)
        x = self.conv5(block_2_act, training=training)
        x = self.conv6(x, training=training)
        x = self.bn6(x, training=training)
        x = layers.ReLU()(x)
        x = self.se3(x)
        block_3_act = layers.Add()([x, res3])

        # Block 4
        res4 = self.residual4(block_3_act, training=training)
        x = self.conv7(block_3_act, training=training)
        x = self.conv8(x, training=training)
        x = self.bn8(x, training=training)
        x = layers.ReLU()(x)
        x = self.se4(x)
        block_4_act = layers.Add()([x, res4])

        # Block 5
        res5 = self.residual5(block_4_act, training=training)
        x = self.conv9(block_4_act, training=training)
        x = self.conv10(x, training=training)
        x = self.bn10(x, training=training)
        x = layers.ReLU()(x)
        x = self.se5(x)
        block_5_act = layers.Add()([x, res5])

        x = self.pooling(block_5_act)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        x = self.dense4(x)

        activations = (block_1_act, block_2_act, block_3_act, block_4_act, block_5_act)
        return (x, activations)


class TumourNetWrapper:
    """Wrapper for keras model to add functions that mimic scikit learn model class structure"""

    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def fit(self, X, y, epochs, lr=0.001):
        self.model.fit(
            X,
            y,
            epochs=epochs,
            callbacks=[
                callbacks.LearningRateScheduler(
                    lambda epoch: lr * (0.95 ** (epoch // 5))
                )
            ],
        )

    def compile(self):
        self.model.compile(
            optimizer="adam",
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def save(self, filename):
        self.model.save(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        model = models.load_model(filename)
        if isinstance(model, Model):
            self.model: Model = model

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def summary(self):
        self.model.summary()

    def get_activations(self, X):
        return self.model.activations_call(X)

    def make_gradcam_heatmap(self, image_array, alpha=0.4):
        # Ensure batch dimension exists
        x = np.expand_dims(image_array, axis=0)

        # Convert to tensor so we can watch it
        img_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)

            preds, activations = self.model.activations_call(img_tensor, training=False)
            last_conv_layer_output = activations[-1]

            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # Gradient of the predicted class with respect to the output feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Vector of mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels by how important they are
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        max_val = tf.math.reduce_max(heatmap)

        # Safety check: avoid division by zero
        if max_val > 0:
            # Normalize the heatmap
            heatmap = tf.maximum(heatmap, 0) / max_val
        else:
            # If heatmap is all zeros, return a zero heatmap
            heatmap = tf.zeros_like(heatmap)

        heatmap_array = heatmap.numpy()

        # get original image (remove batch dimension if present)
        if len(image_array.shape) == 4:
            img = image_array[0]
        else:
            img = image_array

        # Rescale heatmap to image size using cv2
        heatmap_resized = cv2.resize(
            heatmap_array, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        # Apply colormap to heatmap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        heatmap_colored = jet_colors[np.uint8(255 * heatmap_resized)]

        # Create the overlay
        img_normalized = img / 255.0 if img.max() > 1 else img
        superimposed_img = heatmap_colored * alpha + img_normalized * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 1)

        # Get prediction info
        pred_class = int(pred_index)
        pred_conf = float(preds[0][pred_index])

        return img, heatmap_resized, superimposed_img, pred_class, pred_conf

    def make_feature_maps(self, image_array, maps_per_layer):
        # Prepare input
        x = np.expand_dims(image_array, axis=0)

        _, activations = self.model.activations_call(x)

        collective_maps = []

        for activation in activations:
            # Remove batch dimension
            activation = activation[0]

            local_maps = []
            for _ in range(maps_per_layer):
                map_idx = np.random.choice(activation.shape[-1])
                map = activation[:, :, map_idx]
                local_maps.append(map)

            collective_maps.append(local_maps)

        return collective_maps


def make_neural_net_generator(
    image_shape, se=True, use_augmentation=True, weights_file=None
):
    def model_generator():
        dummy_input = np.zeros((1, *image_shape), dtype=np.float32)

        if se:
            model = TumourNetSE(use_augmentation=use_augmentation)
        else:
            model = TumourNet(use_augmentation=use_augmentation)

        _ = model(dummy_input, training=True)
        model = TumourNetWrapper(model)
        model.summary()
        if weights_file is not None:
            model.load_weights(weights_file)

        return model

    return model_generator
