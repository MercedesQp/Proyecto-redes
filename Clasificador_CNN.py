import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import ResNet50


def build_classifier(input_shape=(224, 224, 1), num_classes=6):
    input_img = Input(shape=input_shape)

    # Expandir a 3 canales para que sea compatible con ResNet50
    x = layers.Concatenate()([input_img, input_img, input_img])

    base_model = ResNet50(include_top=False, input_tensor=x, weights='imagenet')
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_img, outputs=output)
    return model


if _name_ == "_main_":
    model = build_classifier()
    model.summary()