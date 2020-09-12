import os
import cv2
import random
import pickle
import pydicom
import numpy as np
from tqdm import tqdm
from scipy import ndimage


def create_training_data(categories, data_path, img_size, training_data):
    for category in categories:

        path = os.path.join(data_path, category)
        class_num = categories.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                for i in range(1):
                    image = pydicom.read_file(path + '/' + img)
                    img_array = np.stack(image.pixel_array)
                    new_array = cv2.resize(img_array, (img_size, img_size))
                    if i == 1:
                        new_array = ndimage.rotate(new_array, 90)
                        training_data.append([new_array, class_num])
                    if i == 2:
                        new_array = ndimage.rotate(new_array, 180)
                        training_data.append([new_array, class_num])
                    else:
                        training_data.append([new_array, class_num])
            except Exception:
                pass


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
    DATA_DIR = 'Data/' + STUDY_TYPE

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
    IMG_SIZE = 100

    training_data = []

    create_training_data(CATEGORIES, DATA_DIR, IMG_SIZE, training_data)
    print('The number of images in the training dataset:', int(len(training_data)))

    random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    out_stream = open('Streams/X.pickle', 'wb')
    pickle.dump(X, out_stream)
    out_stream.close()

    out_stream = open('Streams/y.pickle', 'wb')
    pickle.dump(y, out_stream)
    out_stream.close()


if __name__ == '__main__':
    main()

