from sklearn.metrics import classification_report
from data_augmentation import *
import matplotlib.pyplot as plt
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

BATCH_SIZE = 8
EPOCHS = 50
CLASSES = 2

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(CLASSES, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

# transfer learning
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    epochs=EPOCHS, verbose=1)

predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

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