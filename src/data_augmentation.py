import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from imutils import paths
import numpy as np
import cv2

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

import sys
np.set_printoptions(threshold=sys.maxsize)
matplotlib.use("Agg")

dir = './data'
NORMAL = dir+'/Normal'
COVID = dir+'/COVID-19'
PNEUMONIA = dir+'/PNEUMONIA'


imagePaths1 = list(paths.list_images(NORMAL))
imagePaths2 = list(paths.list_images(COVID))
imagePaths3 = list(paths.list_images(PNEUMONIA))
data = []
labels = []
print('number of images per folder: ', len(imagePaths1), len(imagePaths2), len(imagePaths3))


#loop through images in different folders and append each image and its label to two lists accordingly
def loop_img(imagePaths, dirName):
    for imagePath in imagePaths:
        label = dirName
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        data.append(image)
        labels.append(label)


loop_img(imagePaths1, 'NORMAL')
loop_img(imagePaths2, 'COVID')
# loop_img(imagePaths1, 'PNEUMONIA')
# loop_img(imagePaths1, 'VP')

print('number of images in data:', len(data))
print('number of labels: ', len(set(labels)))

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels, stratify=labels, test_size=0.3, random_state=42)

trainX = trainX.reshape(trainX.shape[0], -1)

sm = SMOTE(random_state=12)
trainX_res, trainY_res = sm.fit_resample(trainX, trainY[:, 0])

clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(trainX_res, trainY_res)

trainY_res = [[i, 1. - i] for i in trainY_res]
trainX_res = trainX_res.reshape(trainX_res.shape[0], 224, 224, 3)

aug = ImageDataGenerator(rotation_range=10,
                         fill_mode="nearest")
