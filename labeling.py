import os
import pydicom
import numpy as np

from matplotlib import pyplot as plt
from pydicom.datadict import DicomDictionary, keyword_dict
from pydicom.pixel_data_handlers.numpy_handler import pack_bits


def load_images(path):
    all_img = [pydicom.read_file(path + '/' + i) for i in os.listdir(path)]
    return all_img


def get_voxels(img):
    image1 = np.stack(img.pixel_array)
    image1 = image1.astype(np.int16)
    image1[image1 == -2000] = 0
    intercept = img.RescaleIntercept
    slope = img.RescaleSlope
    if slope != 1:
        image1 = slope * image1.astype(np.float64)
        image1 = image1.astype(np.int16)
    image1 += np.int16(intercept)
    return np.array(image1, dtype=np.int16)


def standardize_voxel_values(img):
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    return img


def show_standardize_image(img, title='Standardized image'):
    fig, ax = plt.subplots(1, 1, figsize=[10, 10])
    ax.set_title(title)
    ax.imshow(img, cmap='gray')
    ax.axis('on')
    plt.show()


def show_pixel_array(file, title):
    fig, ax = plt.subplots(1, 1, figsize=[10, 10])
    ax.set_title(title)
    ax.imshow(file, cmap='gray')
    ax.axis('on')
    plt.show()


def create_demo_box(img, point_x_beg, point_y_beg, point_x_end, point_y_end):
    img[point_y_beg, point_x_beg:point_x_end] = 1
    img[point_y_end, point_x_beg:point_x_end] = 1
    img[point_y_beg:point_y_end, point_x_beg] = 1
    img[point_y_beg:point_y_end, point_x_end] = 1
    return img


def create_box(img, point_x_beg, point_y_beg, point_x_end, point_y_end):
    bound_box = img
    for i in range(len(bound_box)):
        for j in range(len(bound_box[i])):
            bound_box[i][j] = 0

    bound_box[point_y_beg, point_x_beg:point_x_end] = 1
    bound_box[point_y_end, point_x_beg:point_x_end] = 1
    bound_box[point_y_beg:point_y_end, point_x_beg] = 1
    bound_box[point_y_beg:point_y_end, point_x_end] = 1
    return bound_box


def add_overlay(img, area, array):
    groups = {
        'Head': 0x6000,
        'Neck': 0x6002,
        'Chest': 0x6004,
        'Abdomen': 0x6006,
        'Pelvis': 0x6008,
    }

    string = area + ': Overlay Rows'

    new_dict_items = {
        (groups[area], 0x0010): ('US', '1', string, '', 'OverlayRows'),
        (groups[area], 0x0011): ('US', '1', area + ": Overlay Columns", '', 'OverlayColumns'),
        (groups[area], 0x0015): ('IS', '1', area + ": Number of Frames in Overlay", '', 'NumberFrames'),
        (groups[area], 0x0022): ('LO', '1', area + ": Overlay Description ", '', 'OverlayDescription'),
        (groups[area], 0x0040): ('CS', '1', area + ": Overlay Type", '', 'OverlayType'),
        (groups[area], 0x0050): ('SS', '2', area + ": Overlay Origin", '', 'OverlayOrigin'),
        (groups[area], 0x0051): ('US', '1', area + ": Image Frame Origin ", '', 'ImageFrameOrigin'),
        (groups[area], 0x0100): ('US', '1', area + ": Overlay Bits Allocated", '', 'OverlayBitsAllocated'),
        (groups[area], 0x0102): ('US', '1', area + ": Overlay Bit Position", '', 'OverlayBitPosition'),
        (groups[area], 0x3000): ('OW', '1', area + ": Overlay Data", '', 'OverlayData'),
    }

    DicomDictionary.update(new_dict_items)
    new_names_dict = dict([(val[4], tag) for tag, val in new_dict_items.items()])
    keyword_dict.update(new_names_dict)
    row_count, col_count = array.shape
    img.OverlayRows = row_count
    img.OverlayColumns = col_count
    img.NumberFrames = 1
    img.OverlayDescription = area
    img.OverlayType = 'G'
    img.OverlayOrigin = [1, 1]
    img.ImageFrameOrigin = 1
    img.OverlayBitsAllocated = 1
    img.OverlayBitPosition = 0
    array_new = np.reshape(array, array.size)
    packed_bytes = pack_bits(array_new)

    if len(packed_bytes) % 2:
        packed_bytes += b'\x00'

    img.OverlayData = packed_bytes
    img[groups[area], 0x3000].VR = 'OW'
    return img


def main():
    print("""
Enter the path to directopy with the images you want to annotate:
(My path: '/home/computer/SCIENTIFIC_WORK/segmentation_medical_images/ct_locator_sample')
          """)
    data_path = str(input())
    images = load_images(data_path)

    continue_status = 'y'
    while continue_status == 'y':
        print("""
Select study type for which images are annotated:
* Head detection in the scan: head_segmentation
* Neck detection in the scan: neck_segmentation
* Chest detection in the scan: chest_segmentation
* Abdomen detection in the scan: abdomen_segmentation
* Pelvis detection in the scan: pelvis_segmentation
             """)
        STUDY_TYPE = str(input())

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

        output_path_with = 'Data/' + STUDY_TYPE + '/' + BODY_PART_NAME + '/'
        output_path_without = 'Data/' + STUDY_TYPE + '/Other/'

        print('Enter image number:')
        number = int(input())
        image = images[number]

        image_vox = get_voxels(image)
        image_std = standardize_voxel_values(image_vox)
        show_standardize_image(image_std)

        print('Enter the coordinates of the first point of the bounding box:')
        beg_point_x = int(input())
        beg_point_y = int(input())

        print('Enter the coordinates of the end point of the bounding box:')
        end_point_x = int(input())
        end_point_y = int(input())

        image_st = create_demo_box(image_std, beg_point_x, beg_point_y, end_point_x, end_point_y)
        show_standardize_image(image_st)
        image_for_box = image_std
        bounding_box = create_box(image_for_box, beg_point_x, beg_point_y, end_point_x, end_point_y)

        print('Enter the name of the marked object: (Head, Neck, Chest, Abdomen, Pelvis)')
        part_name = input()
        image = add_overlay(image, part_name, bounding_box)

        if part_name == BODY_PART_NAME:
            image.save_as(output_path_with + 'image_' + str(number) + '.dcm')
        else:
            image.save_as(output_path_without + 'image_' + str(number) + '.dcm')

        print(image)
        print('Continue annotating? Enter y/n:')
        continue_status = str(input())


if __name__ == '__main__':
    main()

