import os
import cv2
import pydicom
import numpy as np
import tensorflow as tf

from tqdm import tqdm


def create_input_data(data_dir):
    IMG_SIZE = 100
    testing_data = []
    path = os.path.join(data_dir)
    for img in tqdm(os.listdir(path)):
        try:
            image = pydicom.read_file(path + '/' + img)
            img_array = np.stack(image.pixel_array)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data.append(new_array)
        except Exception:
            pass

    X = []
    for features in testing_data:
        X.append(features)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X/255.0

    return X


def main():
    print("""
Select study type for which images are annotated:
* Head detection in the scan: head_segmentation
* Neck detection in the scan: neck_segmentation
* Chest detection in the scan: chest_segmentation
* Abdomen detection in the scan: abdomen_segmentation
* Pelvis detection in the scan: pelvis_segmentation
          """)
    STUDY_TYPE = str(input())

    print("""
Specify the directory with the investigated images / investigated images:
(My path: '/home/computer/SCIENTIFIC_WORK/segmentation_medical_images/Images')
          """)
    DATA_DIR = str(input())

    BODY_PART_NAME = ''
    if STUDY_TYPE == 'head_segmentation':
        BODY_PART_NAME = 'Head'
    elif STUDY_TYPE == 'neck_segmentation':
        BODY_PART_NAME = 'Neck'
    elif STUDY_TYPE == 'chest_segmentation':
        BODY_PART_NAME = 'Chest'
    elif STUDY_TYPE == 'abdomen_segmentation':
        BODY_PART_NAME = 'Abdomen'
    elif STUDY_TYPE == 'pelvis_segmentation':
        BODY_PART_NAME = 'Pelvis'

    CATEGORIES = [BODY_PART_NAME, 'Other']

    model = tf.keras.models.load_model('Models/' + STUDY_TYPE + '.model')

    X = create_input_data(DATA_DIR)
    predictions = model.predict(X)

    print('The name of the detected part of the body in the picture:', CATEGORIES[int(predictions[0][0])])


if __name__ == '__main__':
    main()
