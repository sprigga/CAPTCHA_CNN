import numpy as np
import os
import time
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 設定參數
epochs = 50
digits_in_img = 4
x_list = []
y_list = []
img_filenames = os.listdir(r'label_captcha_tool-master/captcha')
labeled = open(r'label_captcha_tool-master/label.csv', 'r')
run_id = time.strftime('Run_Time_%Y_%m_%d_%H_%M_%S')
tensorboard_path = os.path.join('logs', run_id)
dict_captcha = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, 'a': 8, 'b': 9, 'c': 10,
                'd': 11, 'e': 12, 'f': 13, 'g': 14, 'n': 15, 'm': 16, 'p': 17, 'w': 18, 'x': 19, 'y': 20}

def sort_key(s):
    try:
        c = re.findall('^\d+', s)[0]
    except:
        c = -1
    return int(c)

def to_onehot(c):
    return dict_captcha.get(c, 0)

def split_digits_in_img(img_array, x_list, y_list):
    split_label = labeled.readline().strip()
    img_rows, img_cols, _ = img_array.shape
    step = img_cols // digits_in_img
    for i in range(digits_in_img):
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255.0)
        onehot = to_onehot(split_label[i])
        y_list.append(onehot)

def train_dataset(x_list, y_list):
    y_list = tf.keras.utils.to_categorical(y_list, num_classes=21)
    x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.2, random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def train_model(x_train, x_test, y_train, y_test, img_rows, img_cols):
    # 設定模型儲存與回調
    model_path = 'spas_cnn_model_tf2_v3.h5'
    best_model_path = 'best_model_weights.h5'
    final_model_path = 'spas_cnn_model_tf2_v3_final.h5'
    
    tensorboard = TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True,
                                       save_weights_only=False, mode='max', verbose=1)
    callbacks_list = [tensorboard, early_stop, lr_scheduler, model_checkpoint]

    weight_decay = 1e-4  # L2 正則化
    if os.path.exists(model_path):
        model = models.load_model(model_path)
        print('Model loaded from file.')
    else:
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu',
                          kernel_regularizer=regularizers.l2(weight_decay),
                          input_shape=(img_rows, img_cols // digits_in_img, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(weight_decay)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(21, activation='softmax')
        ])
        print('New model created.')

    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # 使用資料增強訓練
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    datagen.fit(x_train)

    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              validation_data=(x_test, y_test),
              epochs=50,
              callbacks=callbacks_list)

    # 評估最佳模型
    best_model = models.load_model(best_model_path)
    loss, accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    print('Best model test loss:', loss)
    print('Best model test accuracy:', accuracy)

    # 儲存最終產品模型
    best_model.save(final_model_path)
    print(f'Final product model saved to {final_model_path}')
    return best_model

if __name__ == '__main__':
    img_rows = None
    img_cols = None

    for img_filename in sorted(img_filenames, key=sort_key):
        if '.jpg' not in img_filename:
            continue
        img = load_img(os.path.join('label_captcha_tool-master', 'captcha', img_filename), color_mode='grayscale')
        img_array = img_to_array(img)
        img_rows, img_cols, _ = img_array.shape
        split_digits_in_img(img_array, x_list, y_list)

    x_train, x_test, y_train, y_test = train_dataset(x_list, y_list)

    # Reshape data for CNN input
    x_train = x_train.reshape(-1, img_rows, img_cols // digits_in_img, 1)
    x_test = x_test.reshape(-1, img_rows, img_cols // digits_in_img, 1)

    train_model(x_train, x_test, y_train, y_test, img_rows, img_cols)

    labeled.close()
