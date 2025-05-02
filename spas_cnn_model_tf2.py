import cv2
import numpy as np
import os
import re
import time
from tensorflow.keras import models, layers, Input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 設定參數
img_rows = 50
img_cols = 88
digits_in_img = 4
img_filenames = os.listdir(r'test_captcha')
dict_captcha = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, 'a': 8, 'b': 9, 'c': 10,
                'd': 11, 'e': 12, 'f': 13, 'g': 14, 'n': 15, 'm': 16, 'p': 17, 'w': 18, 'x': 19, 'y': 20}
np.set_printoptions(suppress=True, linewidth=150, precision=3, formatter={'float': '{: 0.3f}'.format})

def processImg(img_filename):
    img_cv = cv2.imread(img_filename)
    if img_cv is None:
        raise ValueError(f"Failed to load image: {img_filename}")
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    x, y, w, h = [102, 0, 88, 50]
    dst_resize = img_gray[y:y+h, x:x+w]
    # 確保輸出尺寸為 (50, 88)
    dst_resize = cv2.resize(dst_resize, (img_cols, img_rows))
    print("Processing captcha")
    time.sleep(0.5)
    cv2.imwrite(r"test_captcha/processed_captcha.jpg", dst_resize)

def processBatchImg():
    for i in range(1201, 1301):
        img_path = f"getKaptchaImg/getKaptchaImg{i}.jpeg"
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"Failed to load image: {img_path}")
            continue
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        x, y, w, h = [102, 0, 88, 50]
        dst_resize = img_gray[y:y+h, x:x+w]
        # 確保輸出尺寸為 (50, 88)
        dst_resize = cv2.resize(dst_resize, (img_cols, img_rows))
        print(f"Processing captcha_{i}")
        time.sleep(0.5)
        cv2.imwrite(f"test_captcha/{i}.jpg", dst_resize)

def reverse_list(c):
    for key, value in dict_captcha.items():
        if value == c:
            return key
    return None

def sort_key(s):
    try:
        c = re.findall('^\d+', s)[0]
    except:
        c = -1
    return int(c)

def split_digits_in_img(img_array):
    x_list = []
    step = img_cols // digits_in_img
    for i in range(digits_in_img):
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255.0)
    return x_list

def cnn_model_predict(predict_img):
    model_path = 'spas_cnn_model_tf2_v3_final.h5'
    if os.path.exists(model_path):
        model = models.load_model(model_path)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print('No trained model found.')
        exit(-1)

    img = load_img(predict_img, color_mode='grayscale')
    img_array = img_to_array(img)
    if img_array.shape[:2] != (img_rows, img_cols):
        print(f"Resizing image to ({img_rows}, {img_cols})")
        img_array = cv2.resize(img_array, (img_cols, img_rows))
        img_array = img_array.reshape(img_rows, img_cols, 1)
    x_list = split_digits_in_img(img_array)

    verification_code = []
    for i in range(digits_in_img):
        input_data = x_list[i].reshape(1, img_rows, img_cols // digits_in_img, 1)
        confidences = model.predict(input_data, verbose=0)
        result_class = np.argmax(confidences, axis=1)
        result_value = reverse_list(result_class[0])
        verification_code.append(result_value)
        print(f'Digit {i + 1}: Confidence=> {np.squeeze(confidences)}    Predict=> {np.squeeze(result_class)}')
    print('Predicted verification code:', verification_code)
    print('\r\n')
    return verification_code

def cnn_model_batch_predict():
    model_path = 'spas_cnn_model_tf2_v3_final.h5'
    if os.path.exists(model_path):
        model = models.load_model(model_path)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print('No trained model found.')
        exit(-1)

    for img_filename in sorted(img_filenames, key=sort_key):
        print(img_filename)
        if '.jpg' not in img_filename:
            continue
        img_path = os.path.join("test_captcha", img_filename)
        img = load_img(img_path, color_mode='grayscale')
        img_array = img_to_array(img)
        if img_array.shape[:2] != (img_rows, img_cols):
            print(f"Resizing image {img_filename} to ({img_rows}, {img_cols})")
            img_array = cv2.resize(img_array, (img_cols, img_rows))
            img_array = img_array.reshape(img_rows, img_cols, 1)
        x_list = split_digits_in_img(img_array)

        verification_code = []
        for i in range(digits_in_img):
            input_data = x_list[i].reshape(1, img_rows, img_cols // digits_in_img, 1)
            confidences = model.predict(input_data, verbose=0)
            result_class = np.argmax(confidences, axis=1)
            result_value = reverse_list(result_class[0])
            verification_code.append(result_value)
            print(f'Digit {i + 1}: Confidence=> {np.squeeze(confidences)}    Predict=> {np.squeeze(result_class)}')
        print('Predicted verification code:', verification_code)
        print('\r\n')
        time.sleep(0.01)

def captcha_code(img_filename, predict_img):
    processImg(img_filename)
    time.sleep(0.01)
    captcha_list = cnn_model_predict(predict_img)
    captcha_str = "".join(captcha_list)
    print("Captcha code:", captcha_str)
    return captcha_str

if __name__ == '__main__':
    # img_filename = r'getKaptchaImg/getKaptchaImg1400.jpeg'
    # predict_img = r'test_captcha/processed_captcha.jpg'
    # captcha_code(img_filename, predict_img)
    cnn_model_batch_predict()