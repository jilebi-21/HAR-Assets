import os, cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(rotation_range=10, brightness_range=[.2,1.25])
image_path = "C:/Users/Pruthvi/Documents/project/test"
cou = 5 #no of images to be generated


def get_extension(fileName):
    if(fileName.find('.')):
        return fileName[fileName.rindex('.') + 1:len(fileName)]
    return ''


for i in os.listdir(image_path):
    for d in os.listdir(image_path + '/' + i):
        path = image_path + '/' + i + '/' + d
        img = cv2.imread(path)
    
        it = datagen.flow(np.expand_dims(img, axis=0), batch_size=1)
        for l in range(cou):
            batch = it.next()
            image = batch[0].astype('uint8')
            file_name = d[0:d.rindex('.')] + '__' + str(l + 1) + '_.' + get_extension(d)
            cv2.imwrite(image_path + '/' + i + '/' + file_name, image)
