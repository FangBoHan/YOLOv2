from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, concatenate, Reshape, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf


def get_yolov2(image_w, image_h, box_num, class_count):
    """
    Build a model of YOLOv2.
    Parameters use PASCAL VOC by default.
    
    Parameters
    ----------
    - image_w : int, width of images
    - image_h : height of images
    - box_num : number of anchor box
    - class_count : number of class
    
    Returns
    -------
    - model : tensorflow model, yolov2 model
    """
    # 建立模型框架
    grid_w, grid_h = image_w // 32, image_h // 32
    
    input_image = Input(shape=(image_w, image_h, 3))
    
    x = Conv2D(32, (3,3), padding="same", use_bias=False, name="conv_1")(input_image)
    x = BatchNormalization(name="norm_1")(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(64, (3,3), padding="same", use_bias=False, name="conv_2")(x)
    x = BatchNormalization(name="norm_2")(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(128, (3,3), padding="same", use_bias=False, name="conv_3")(x)
    x = BatchNormalization(name="norm_3")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, (1,1), padding="same", use_bias=False, name="conv_4")(x)
    x = BatchNormalization(name="norm_4")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (3,3), padding="same", use_bias=False, name="conv_5")(x)
    x = BatchNormalization(name="norm_5")(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(256, (3,3), padding="same", use_bias=False, name="conv_6")(x)
    x = BatchNormalization(name="norm_6")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (1,1), padding="same", use_bias=False, name="conv_7")(x)
    x = BatchNormalization(name="norm_7")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (3,3), padding="same", use_bias=False, name="conv_8")(x)
    x = BatchNormalization(name="norm_8")(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(512, (3,3), padding="same", use_bias=False, name="conv_9")(x)
    x = BatchNormalization(name="norm_9")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (1,1), padding="same", use_bias=False, name="conv_10")(x)
    x = BatchNormalization(name="norm_10")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (3,3), padding="same", use_bias=False, name="conv_11")(x)
    x = BatchNormalization(name="norm_11")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (1,1), padding="same", use_bias=False, name="conv_12")(x)
    x = BatchNormalization(name="norm_12")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (3,3), padding="same", use_bias=False, name="conv_13")(x)
    x = BatchNormalization(name="norm_13")(x)
    x = LeakyReLU(0.1)(x)
    
    passthrough = x
    
    x = MaxPooling2D()(x)
    
    x = Conv2D(1024, (3,3), padding="same", use_bias=False, name="conv_14")(x)
    x = BatchNormalization(name="norm_14")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (1,1), padding="same", use_bias=False, name="conv_15")(x)
    x = BatchNormalization(name="norm_15")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, (3,3), padding="same", use_bias=False, name="conv_16")(x)
    x = BatchNormalization(name="norm_16")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (1,1), padding="same", use_bias=False, name="conv_17")(x)
    x = BatchNormalization(name="norm_17")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, (3,3), padding="same", use_bias=False, name="conv_18")(x)
    x = BatchNormalization(name="norm_18")(x)
    x = LeakyReLU(0.1)(x)
    
    x = Conv2D(1024, (3,3), padding="same", use_bias=False, name="conv_19")(x)
    x = BatchNormalization(name="norm_19")(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, (3,3), padding="same", use_bias=False, name="conv_20")(x)
    x = BatchNormalization(name="norm_20")(x)
    x = LeakyReLU(0.1)(x)
    
    passthrough = Conv2D(64, (1, 1), padding="same", use_bias=False, name="conv_21")(passthrough)
    passthrough = BatchNormalization(name="norm_21")(passthrough)
    passthrough = LeakyReLU(0.1)(passthrough)
    passthrough = tf.nn.space_to_depth(passthrough, 2)    # 先經過一層 conv. layer 後再做 passthrough layer 處理
    
    x = concatenate([passthrough, x])
    
    x = Conv2D(1024, (3,3), padding="same", use_bias=False, name="conv_22")(x)
    x = BatchNormalization(name="norm_22")(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.3)(x)    # 參考別人的作法，這裡要加 dropout
    
    x = Conv2D(box_num * (4 + 1 + class_count), (1,1), padding="same", name="conv_23")(x)
    output = Reshape((grid_w, grid_h, box_num, 4 + 1 + class_count))(x)
    
    model = Model(input_image, output)
    
    return model

    
    
    




