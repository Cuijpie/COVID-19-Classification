import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from keras.utils import np_utils
from imutils import paths
import numpy as np
import cv2

matplotlib.use("Agg")

dir = './data'
Normal = dir + '/Normal'
Covid = dir + '/COVID-19'
BP = dir + '/BacterialPneumonia'
VP = dir + '/ViralPneumonia'

imagePaths1 = list(paths.list_images(Normal))
imagePaths2 = list(paths.list_images(Covid))
imagePaths3 = list(paths.list_images(BP))
imagePaths4 = list(paths.list_images(VP))
data = []
labels = []
print('number of images per folder: ', len(imagePaths1), len(imagePaths2), len(imagePaths3), len(imagePaths4))


# loop through images in different folders and append each image and its label to two lists accordingly
def loop_img(imagePaths, dirName):
    for imagePath in imagePaths:
        label = dirName
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (224, 224))  # image risize
        data.append(image)
        labels.append(label)

loop_img(imagePaths1, 'Normal')
loop_img(imagePaths2, 'COVID')
loop_img(imagePaths1, 'BP')
loop_img(imagePaths1, 'VP')

print('number of images in data:', len(data))
print('number of labels: ', len(set(labels)))

data = np.array(data, dtype="float") / 255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 4)
print(data.shape[1:], labels.shape[0])

data_gen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img(Covid + '/3.jpeg')  # this is a PIL image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0

for batch in data_gen.flow(x, batch_size=1, save_to_dir=Covid, save_prefix='covid', save_format='jpeg'):
    i += 1

    if i > 300:
        break


# (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.3, random_state=42)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        './data',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        './val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

generator = data_gen.flow_from_directory(
        './val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',  # only data, no labels
        shuffle=False)  # keep data in same order as labels