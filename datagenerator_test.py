import os
from keras.preprocessing.image import ImageDataGenerator,  img_to_array, load_img
import matplotlib.pyplot as plt

import random
import numpy as np
import cv2 as cv
import imutils

def generate_plot_pics(datagen,orig_img):
    dir_augmented_data = './../dataaugmentation/'
    try:
        ## if the preview folder does not exist, create
        os.mkdir(dir_augmented_data)
    except:
        ## if the preview folder exists, then remove
        ## the contents (pictures) in the folder
        for item in os.listdir(dir_augmented_data):
            os.remove(dir_augmented_data + "/" + item)

    ## convert the original image to array
    x = img_to_array(orig_img)
    ## reshape (Sampke, Nrow, Ncol, 3) 3 = R, G or B
    x = x.reshape((1,) + x.shape)
    ## -------------------------- ##
    ## randomly generate pictures
    ## -------------------------- ##
    i = 0
    Nplot = 8
    for batch in datagen.flow(x,batch_size=1,
                          save_to_dir=dir_augmented_data,
                          save_prefix="pic",
                          save_format='jpeg'):
        i += 1
        if i > Nplot - 1: ## generate 8 pictures
            break

    ## -------------------------- ##
    ##   plot the generated data
    ## -------------------------- ##
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(hspace=0.02,wspace=0.01,
                    left=0,right=1,bottom=0, top=1)

    ## original picture
    ax = fig.add_subplot(3, 3, 1,xticks=[],yticks=[])
    ax.imshow(orig_img)
    ax.set_title("original")

    i = 2
    for imgnm in os.listdir(dir_augmented_data):
        ax = fig.add_subplot(3, 3, i,xticks=[],yticks=[])
        img = load_img(dir_augmented_data + "/" + imgnm)
        ax.imshow(img)
        i += 1
    plt.show()

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 15
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

def add_peoples(img):
    '''Add random noise to an image'''

    directory_people = "./../raw_datasets_images/obstruction/people/processed"
    onlyfiles = [f for f in os.listdir(directory_people) if os.path.isfile(os.path.join(directory_people, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread(os.path.join(directory_people, onlyfiles[n]))
    img3 = np.array(img)
    #print(img.width*0.2, img.height*0.2)
    #print(images[0].shape[1], images[0].shape[0])
    h, w = images[0].shape[:2]
    pip_h = 10
    pip_w = 10
    aux = cv.cvtColor(images[0], cv.COLOR_BGR2RGB)
    #aux = rotate_bound(aux, 45)
    #h, w = aux.shape[:2]
    img3[pip_h:pip_h + h, pip_w:pip_w + w] = np.where(aux == 255, img3[pip_h:pip_h + h, pip_w:pip_w + w], aux)
    #cv.imshow("test", img3)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    #np.clip(img, 0., 255.)
    return img3

orig_img = load_img("./../raw_datasets_images/2020_06_25-14_14_59/2732.jpg")

## rotation_range: Int. Degree range for random rotations.
#rotation_range=20
#width_shift_range=0.2
#height_shift_range=0.2
#zoom_range=0.2
#preprocessing_function=add_noise
datagen = ImageDataGenerator(preprocessing_function=add_peoples)
generate_plot_pics(datagen, orig_img)

#a = add_peoples(orig_img, 2, 2)
