
from sklearn.metrics import classification_report, confusion_matrix
from data_augmentation import *
from keras.optimizers import Adam
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Flatten
from keras.models import Model
from keras import backend as K


BATCH_SIZE = 16
EPOCHS = 50

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# transfer learning
for layer in base_model.layers:
    layer.trainable = False

opt = Adam(lr=0.0001)

model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['acc', f1_m, precision_m, recall_m])

# train the network
print("[INFO] training network...")

class_weight = {
    0: 95,
    1: 5
}

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BATCH_SIZE,
                        epochs=EPOCHS,
                        class_weight=class_weight)


model.save_weights('test_inception.h5')

#model.load_weights('test_inception.h5')

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