import matplotlib
from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array

matplotlib.use("Agg")

dir = './data'
Normal = dir + '/Normal'
Covid = dir + '/COVID-19'

data_gen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#for imgPath in imagePaths2:
#    img = load_img(imgPath)
#    x = img_to_array(img)
#    x = x.reshape((1,) + x.shape)
#    i = 0
#    for batch in data_gen.flow(x, batch_size=1, save_to_dir=Covid, save_prefix='aug', save_format='jpeg'):
#        i += 1
#        if i > 10:
#            break

train_datagen = ImageDataGenerator(
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
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        './val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

generator = data_gen.flow_from_directory(
        './val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)