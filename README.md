# CAPTCHA CNN Recognition System

This project is a CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) automatic recognition system based on TensorFlow/Keras. It includes two main modules, training and prediction, and supports batch processing.
This README, based on the source code and training record files, explains the project structure, training process, model performance, and usage.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Functionality Description](#functionality-description)
- [Training Process and Model Performance Analysis](#training-process-and-model-performance-analysis)
- [Prediction Process](#prediction-process)
- [UML Communication Diagram](#uml-communication-diagram)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)
- [Frequently Asked Questions](#frequently-asked-questions)

---

## Project Structure

```
captcha/
├── spas_train_tf2.py                # Main program for model training
├── spas_cnn_model_tf2.py            # Main program for model prediction
├── train_data/                      # Training process records (json)
│   ├── Run_Time_2025_05_02_15_53_10_train.json           # Training accuracy
│   ├── Run_Time_2025_05_02_15_53_10_train_loss.json      # Training loss
│   ├── Run_Time_2025_05_02_15_53_10_train_lr.json        # Learning rate
│   ├── Run_Time_2025_05_02_15_53_10_validation.json      # Validation accuracy
│   └── Run_Time_2025_05_02_15_53_10_validation_loss.json # Validation loss
├── label_captcha_tool-master/
│   ├── captcha/                     # Original CAPTCHA images for training
│   └── label.csv                    # CAPTCHA labels
├── test_captcha/                    # Images for testing/prediction
├── logs/                            # TensorBoard log files
├── spas_cnn_model_tf2_v3.h5         # Model saved during training (may be overwritten)
├── best_model_weights.h5            # Model weights with the best validation accuracy
└── spas_cnn_model_tf2_v3_final.h5   # Final model for prediction (from best_model)
```

---

## Functionality Description

- **spas_train_tf2.py**
  - Reads images from `label_captcha_tool-master/captcha` and labels from `label.csv`.
  - Preprocesses images (grayscale, character segmentation, normalization).
  - Converts labels to One-Hot encoding.
  - Splits data into training and validation sets.
  - Builds or loads a CNN model.
  - Trains the model using data augmentation, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint strategies.
  - Records training metrics (Accuracy, Loss, Learning Rate) in the `train_data/` folder.
  - Saves the model with the highest validation accuracy (`best_model_weights.h5`) and also saves it as the final model (`spas_cnn_model_tf2_v3_final.h5`).

- **spas_cnn_model_tf2.py**
  - Provides `processImg` and `processBatchImg` functions to preprocess original CAPTCHA images (cropping, grayscale, scaling).
  - Provides `cnn_model_predict` function to predict a single preprocessed image.
  - Provides `cnn_model_batch_predict` function to batch predict all images in the `test_captcha` folder.
  - Loads the `spas_cnn_model_tf2_v3_final.h5` model for prediction.
  - Outputs the confidence level (Softmax output) for each character and the final predicted CAPTCHA string.

---

## Training Process and Model Performance Analysis

### Training Process

1.  **Data Preparation**: Reads images and labels from specified paths, ensures order using `sort_key`, splits each image into 4 characters using `split_digits_in_img`, and normalizes them. Converts labels using `to_onehot`.
2.  **Dataset Splitting**: Splits data into 80% training set and 20% validation set using `train_test_split`.
3.  **Model Architecture**:
    - `Conv2D(32, (3,3), relu)` + `BatchNormalization`
    - `Conv2D(64, (3,3), relu)` + `BatchNormalization`
    - `MaxPooling2D(2,2)`
    - `Dropout(0.3)`
    - `Flatten`
    - `Dense(128, relu)` + `BatchNormalization`
    - `Dropout(0.4)`
    - `Dense(21, softmax)` (corresponding to 20 characters + 1 unknown/background in `dict_captcha`)
    - Uses L2 regularization (`weight_decay = 1e-4`).
4.  **Training Strategies**:
    - **Optimizer**: Adam (initial learning rate 0.001)
    - **Loss Function**: Categorical Crossentropy
    - **Data Augmentation**: `ImageDataGenerator` (rotation, translation, scaling, shearing)
    - **Callbacks**:
        - `TensorBoard`: Records training process for visualization.
        - `EarlyStopping`: Monitors `val_accuracy`, stops training if no improvement for 5 epochs, restores best weights.
        - `ReduceLROnPlateau`: Monitors `val_accuracy`, halves learning rate if no improvement for 3 epochs.
        - `ModelCheckpoint`: Saves model weights with the highest `val_accuracy` to `best_model_weights.h5`.
    - **Epochs**: Set to 50 but may end early due to EarlyStopping.

### Training Record Analysis (Based on Run_Time_2025_05_02_15_53_10)

- **Training Cycles**: 27 Epochs (0-26).
- **Accuracy**:
    - **Training Set**: Improved from ~31.9% to ~94.4%.
    - **Validation Set**: Started at ~4.9%, reached ~97.5% at epoch 7, stabilized at ~98.6%.
- **Loss**:
    - **Training Set**: Decreased from ~2.52 to ~0.31.
    - **Validation Set**: Started at ~4.14, stabilized between 0.16-0.24 after epoch 7, lowest at ~0.167.
- **Learning Rate**:
    - Epoch 0-10: 0.001
    - Epoch 11-17: 0.0005 (first reduction)
    - Epoch 18-24: 0.00025 (second reduction)
    - Epoch 25-26: 0.000125 (third reduction)

#### Learning Curve Summary

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Learning Rate |
| :---- | :-------- | :------ | :--------- | :------- | :------------ |
| 0     | 0.319     | 0.049   | 2.516      | 4.140    | 0.001         |
| 7     | 0.870     | 0.975   | 0.547      | 0.233    | 0.001         |
| 11    | 0.903     | 0.981   | 0.463      | 0.205    | 0.0005        |
| 18    | 0.923     | 0.985   | 0.367      | 0.173    | 0.00025       |
| 25    | 0.941     | 0.986   | 0.326      | 0.167    | 0.000125      |
| 26    | 0.944     | 0.986   | 0.314      | 0.173    | 0.000125      |

- **Conclusion**: The model performs well, with high and stable validation accuracy. The learning rate adjustment strategy is effective, and there is no significant overfitting. The best model appears at epoch 25 or 26.

---

## Prediction Process

1.  **Load Model**: Load `spas_cnn_model_tf2_v3_final.h5`.
2.  **Image Preprocessing**:
    - Read image (`load_img`).
    - Convert to NumPy array (`img_to_array`).
    - Resize using `cv2.resize` if dimensions do not match `(img_rows, img_cols)`.
    - Ensure grayscale single channel.
3.  **Character Segmentation**: Use `split_digits_in_img` to split the preprocessed image into 4 character sub-images and normalize them.
4.  **Model Prediction**:
    - Predict each character sub-image using `model.predict`.
    - Use `np.argmax` to find the class index with the highest probability.
    - Convert index back to corresponding character using `reverse_list`.
5.  **Output Results**: Print confidence level (Softmax output) and predicted class for each character, then combine and output the predicted CAPTCHA string.

---

## UML Communication Diagram

The following diagram (CAPTCHA_CNN.png) is the UML communication diagram for the project's "Batch CAPTCHA Prediction" process, visualizing the interaction between main objects:

- **User** triggers `spas_cnn_model_tf2.py` to execute batch prediction.
- The main program loads the trained model and retrieves all images to be predicted from the `test_captcha/` directory.
- Each image undergoes preprocessing (grayscale, cropping, scaling) before being sent to the model for prediction.
- Prediction results (confidence level for each character and final CAPTCHA string) are compiled and output to the user.

![CAPTCHA_CNN UML Communication Diagram](CAPTCHA_CNN.png)

This diagram helps understand the collaboration between program modules and objects during batch prediction.

---

## How to Use

### 1. Prepare Environment and Data

- Install dependencies (see next section).
- Place training images in `label_captcha_tool-master/captcha/`.
- Place training labels in `label_captcha_tool-master/label.csv` (must correspond to image filenames in order).
- Place images to be predicted in `test_captcha/` (for batch prediction) or other specified paths (for single image prediction).

### 2. Train Model

Run the following command in the terminal:

```bash
python spas_train_tf2.py
```

After training, the best model will be saved as spas_cnn_model_tf2_v3_final.h5.

### 3. Batch Predict Images

Modify the `if __name__ == '__main__':` block in spas_cnn_model_tf2.py:

```python
if __name__ == '__main__':
    # Ensure processBatchImg() or other preprocessing functions have placed images in test_captcha
    cnn_model_batch_predict()
```

Then run the following command in the terminal:

```bash
python spas_cnn_model_tf2.py
```

### 4. Predict Single Image

Modify the `if __name__ == '__main__':` block in spas_cnn_model_tf2.py:

```python
if __name__ == '__main__':
    img_filename = r'getKaptchaImg/getKaptchaImg1400.jpeg' # Path to original image
    predict_img = r'test_captcha/processed_captcha.jpg' # Path to save preprocessed image
    captcha_code(img_filename, predict_img) # Calls processImg to preprocess the image
```

Then run the following command in the terminal:

```bash
python spas_cnn_model_tf2.py
```

---

## Dependencies

- numpy
- opencv-python
- tensorflow
- scikit-learn

Install using pip:

```bash
pip install numpy opencv-python tensorflow scikit-learn
```

---

## Frequently Asked Questions

### Q1: Warning `Compiled the loaded model, but the compiled metrics have yet to be built` during prediction?

**A:** This occurs because the model is loaded and used for prediction without `compile`. It does not affect prediction, but `model.compile(...)` has been added to the code to remove this warning.

### Q2: Message `Could not identify NUMA node` during training or prediction?

**A:** This is an informational message from TensorFlow when using GPU on Mac, indicating no support for NUMA architecture. It does not affect functionality and can be ignored.

### Q3: How to change the character set or length of the CAPTCHA?

**A:**
- **Character Set**: Modify the `dict_captcha` dictionary in spas_train_tf2.py and spas_cnn_model_tf2.py, and ensure training labels match it. Adjust the output units of the final Dense layer in the model (currently 21).
- **Length**: Modify the `digits_in_img` variable in spas_train_tf2.py and spas_cnn_model_tf2.py. This affects image segmentation and prediction loop count.

---

## Contact and Contribution

For questions or suggestions, feel free to open an Issue or Pull Request.
`````
