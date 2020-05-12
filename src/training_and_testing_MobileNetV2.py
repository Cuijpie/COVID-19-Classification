from data_augmentation import *
from models.googleNet import GoogleNet
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential

# input=tf.keras.Input(shape=data.shape, dtype="float")#input layer with the shape of the training data, currently it's (940,224,224,3)(940 images of 64x64 pixels in RGB (3 channels)))
EPOCHS=30
BatchSize=8
# opt=keras.optimizers.Adam(learning_rate=0.01)
opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2 / EPOCHS)

#MobileNet
base_model = MobileNetV2(input_shape =  data.shape[1:],
                                 include_top = False, weights = None)
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(labels.shape[1], activation = 'sigmoid'))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BatchSize), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BatchSize, epochs=EPOCHS)

predictions = model.predict(testX, batch_size=BatchSize)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("MobileNetV2_training_process.png")
