from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from data_augmentation import *
from keras.optimizers import SGD
from keras.applications import InceptionResNetV2
from keras.applications import InceptionV3
from keras.applications import DenseNet201
from keras.applications import ResNet101V2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.models import Model
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

BATCH_SIZE = 8
EPOCHS = 1
INIT_LR = 0.0001

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(x)
x = BatchNormalization()(x)
predictions = Dense(units=2,
                    activity_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                    activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# transfer learning
for layer in base_model.layers:
    layer.trainable = False

opt = SGD(lr=INIT_LR, momentum=0.5)

model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=['acc'])

# train the network
print("[INFO] training network...")

H = model.fit_generator(aug.flow(trainX_res, trainY_res, batch_size=BATCH_SIZE),
                        validation_data=(testX, testY),
                        epochs=EPOCHS,
                        callbacks=[
                            ModelCheckpoint('XXX.h5',
                                            monitor='val_acc',
                                            save_best_only=True)])

model.load_weights('XXX.h5')

predictions = model.predict(testX, batch_size=BATCH_SIZE)

print(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1)))
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

acc = H.history['acc']
val_acc = H.history['val_acc']
loss = H.history['loss']
val_loss = H.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('<Model>: Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('<name>' + '.png')

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('<Model>: Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('<name>' + '.png')
