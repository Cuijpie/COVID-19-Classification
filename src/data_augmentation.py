import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from imutils import paths
import numpy as np
import cv2


matplotlib.use("Agg")

dir = './data'
Normal=dir+'/Normal'
COVID=dir+'/COVID-19'
BP=dir+'/BacterialPneumonia'
VP=dir+'/ViralPneumonia'

imagePaths1 = list(paths.list_images(Normal))
imagePaths2 = list(paths.list_images(COVID))
imagePaths3 = list(paths.list_images(BP))
imagePaths4 = list(paths.list_images(VP))
data = []
labels = []
print('number of images per folder: ',len(imagePaths1),len(imagePaths2),len(imagePaths3),len(imagePaths4))

#loop through images in different folders and append each image and its label to two lists accordingly
def loop_img(imagePaths, dirName):
    for imagePath in imagePaths:
        label = dirName
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (224, 224))#image risize
        data.append(image)
        labels.append(label)

loop_img(imagePaths1, 'Normal')
loop_img(imagePaths2, 'COVID')
# loop_img(imagePaths1, 'BP')
# loop_img(imagePaths1, 'VP')

print('number of images in data:',len(data))
print('number of labels: ',len(labels))

data = np.array(data, dtype="float") / 255.0
le=LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)
print(data.shape[1:], labels.shape[0])
# print(data[0])
# print(labels[0])

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.3, random_state=42)

aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")


