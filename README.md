# Medical Images Segmentation
This repository describes the method and presents scripts for the segmentation of the sections of the human body in the images of the localizer of a computed tomograph.

## Creating and installing virtual environment
1. I suggest creating an environment from an [environment.yml](https://github.com/AlexeyPopov1997/MedicalImagesSegmentation/blob/master/environment.yml) file (**Warning!!! You need to change `prefix` in the file**):

`conda env create -f environment.yml`

The first line of the `yml` file sets the new environment's name.

2. Activate the new environment:`conda activate med_img_segmentation`

3. Verify that the new environment was installed correctly: `conda env list`

## View a description of the segmentation method
The [med_img_segmentation.ipynb](https://github.com/AlexeyPopov1997/MedicalImagesSegmentation/blob/master/med_img_segmentation.ipynb) file describes how to work with the repository.

To work with this file, you need to add a new kernel to Jupiter Notebook:

`python -m ipykernel install --user --name=med_img_segmentation`
