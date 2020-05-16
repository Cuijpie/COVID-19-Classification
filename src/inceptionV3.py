
from sklearn.metrics import classification_report, confusion_matrix
from data_augmentation import *
from keras.optimizers import Adam
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Flatten
from keras.models import Model
import numpy as np

BATCH_SIZE = 64
EPOCHS = 50

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# transfer learning
for layer in base_model.layers:
    layer.trainable = False

opt = Adam(lr=0.001)

model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // BATCH_SIZE,
    epochs=EPOCHS,#
    validation_data=validation_generator,
    validation_steps=800 // BATCH_SIZE)

model.save_weights('test_inception.h5')

#model.load_weights('test_inception.h5')

probabilities = model.predict_generator(generator, 1607)

y_true = validation_generator.classes
y_pred = np.argmax(probabilities, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
print(cm)
print('Classification Report')
print(classification_report(y_true, y_pred, target_names=['COVID', 'NORMAL', 'PNEUMONIA']))

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