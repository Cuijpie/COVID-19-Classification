
from sklearn.metrics import classification_report, confusion_matrix
from data_augmentation import *
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Flatten
from keras.models import Model

#8
BATCH_SIZE = 8
EPOCHS = 50

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
#0.7
x = Dropout(0.7)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# transfer learning
for layer in base_model.layers:
    layer.trainable = False

#sgd
opt = Adam()

model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['acc'])

# train the network
print("[INFO] training network...")

# 0: 50, 1: 10
class_weight = {
    0: 50,
    1: 1
}

H = model.fit_generator(aug.flow(trainX_res, trainY_res, batch_size=BATCH_SIZE),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BATCH_SIZE,
                        epochs=EPOCHS,
                        class_weight=class_weight)


model.save_weights('test_inception.h5')

#model.load_weights('86-98f1_inception.h5')

predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

#plt.figure()
#plt.ylabel("Loss (training and validation)")
#plt.xlabel("Training Steps")
#plt.ylim([0, 2])
#plt.plot(H["loss"])
#plt.plot(H["val_loss"])

#plt.figure()
#plt.ylabel("Accuracy (training and validation)")
#plt.xlabel("Training Steps")
#plt.ylim([0, 1])
#plt.plot(H["accuracy"])
#plt.plot(H["val_accuracy"])
#plt.savefig('fig.png')