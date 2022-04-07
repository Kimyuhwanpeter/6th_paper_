# -*- coding:utf-8 -*-
import tensorflow as tf

def proposed_block(original_input, encoder_input, decoder_input, filters):

    h = tf.keras.layers.Conv2D(filters=3, kernel_size=1)(encoder_input)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    original_input = tf.image.resize(original_input, [h.shape[1], h.shape[2]])
    h = original_input * tf.nn.sigmoid(h)

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.nn.sigmoid(h)

    out = h * decoder_input

    out = tf.concat([out, decoder_input], -1)
    
    return out

def modified_network(input_shape=(512, 512, 3), nclasses=2):

    backbone = tf.keras.applications.VGG16(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )

    h = inputs = tf.keras.Input(input_shape)

    h = tf.pad(h, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal", name="conv1")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal", name="conv2")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    block_1 = h
    h = tf.keras.layers.MaxPool2D((2,2), strides=2)(h)

    h = tf.pad(h, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, kernel_initializer="he_normal", name="conv3")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, kernel_initializer="he_normal", name="conv4")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    block_2 = h
    h = tf.keras.layers.MaxPool2D((2,2), strides=2)(h)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, kernel_initializer="he_normal", name="conv5")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, kernel_initializer="he_normal", name="conv6")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, kernel_initializer="he_normal", name="conv7")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    block_3 = h
    h = tf.keras.layers.MaxPool2D((2,2), strides=2)(h)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_normal", name="conv8")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_normal", name="conv9")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_normal", name="conv10")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    block_4 = h
    h = tf.keras.layers.MaxPool2D((2,2), strides=2)(h)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_normal", name="conv11")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_normal", name="conv12")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_normal", name="conv13")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    block_5 = h
    h = tf.keras.layers.MaxPool2D((2,2), strides=2)(h)
    #########################################################################################################################
    h_ = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, kernel_initializer="he_normal")(h[:, :, :, 0:384])
    h = tf.keras.layers.UpSampling2D((2,2))(h[:, :, :, 384:])
    h = tf.concat([h, h_], -1)

    h = proposed_block(inputs, block_5, h, 256)
    h = tf.pad(h, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_ = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, kernel_initializer="he_normal")(h[:, :, :, 0:384])
    h = tf.keras.layers.UpSampling2D((2,2))(h[:, :, :, 384:])
    h = tf.concat([h, h_], -1)

    h = proposed_block(inputs, block_4, h, 256)
    h = tf.pad(h, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_ = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, kernel_initializer="he_normal")(h[:, :, :, 0:192])
    h = tf.keras.layers.UpSampling2D((2,2))(h[:, :, :, 192:])
    h = tf.concat([h, h_], -1)

    h = proposed_block(inputs, block_3, h, 128)
    h = tf.pad(h, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_ = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, kernel_initializer="he_normal")(h[:, :, :, 0:96])
    h = tf.keras.layers.UpSampling2D((2,2))(h[:, :, :, 96:])
    h = tf.concat([h, h_], -1)

    h = proposed_block(inputs, block_2, h, 64)
    h = tf.pad(h, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_ = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2, kernel_initializer="he_normal")(h[:, :, :, 0:48])
    h = tf.keras.layers.UpSampling2D((2,2))(h[:, :, :, 48:])
    h = tf.concat([h, h_], -1)

    h = proposed_block(inputs, block_1, h, 32)
    h = tf.pad(h, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1, kernel_initializer="he_normal")(h)

    model = tf.keras.Model(inputs=inputs, outputs=h)

    model.get_layer("conv1").set_weights(backbone.get_layer("block1_conv1").get_weights())
    model.get_layer("conv2").set_weights(backbone.get_layer("block1_conv2").get_weights())
    model.get_layer("conv3").set_weights(backbone.get_layer("block2_conv1").get_weights())
    model.get_layer("conv4").set_weights(backbone.get_layer("block2_conv2").get_weights())
    model.get_layer("conv5").set_weights(backbone.get_layer("block3_conv1").get_weights())
    model.get_layer("conv6").set_weights(backbone.get_layer("block3_conv2").get_weights())
    model.get_layer("conv7").set_weights(backbone.get_layer("block3_conv3").get_weights())
    model.get_layer("conv8").set_weights(backbone.get_layer("block4_conv1").get_weights())
    model.get_layer("conv9").set_weights(backbone.get_layer("block4_conv2").get_weights())
    model.get_layer("conv10").set_weights(backbone.get_layer("block4_conv3").get_weights())
    model.get_layer("conv11").set_weights(backbone.get_layer("block5_conv1").get_weights())
    model.get_layer("conv12").set_weights(backbone.get_layer("block5_conv2").get_weights())
    model.get_layer("conv13").set_weights(backbone.get_layer("block5_conv3").get_weights())


    return model
