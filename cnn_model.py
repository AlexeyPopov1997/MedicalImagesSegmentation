import pickle

from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

print("""
Select study type:
* Head detection in the scan: head_segmentation
* Neck detection in the scan: neck_segmentation
* Chest detection in the scan: chest_segmentation
* Abdomen detection in the scan: abdomen_segmentation
* Pelvis detection in the scan: pelvis_segmentation
      """)
STUDY_TYPE = str(input())

stream_in = open('Streams/' + STUDY_TYPE + '/X.pickle', 'rb')
X = pickle.load(stream_in)

stream_in = open('Streams/' + STUDY_TYPE + '/y.pickle', 'rb')
y = pickle.load(stream_in)

X = X/255.0

# Setting CNN
EPOCHS = 5
DENSE_LAYERS = 0
LAYER_SIZE = 256
CONV_LAYERS = 3
BATCH_SIZE = 32

model = Sequential()

model.add(Conv2D(LAYER_SIZE, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

for layer in range(CONV_LAYERS - 1):
    model.add(Conv2D(LAYER_SIZE, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

for _ in range(DENSE_LAYERS):
    model.add(Dense(LAYER_SIZE))
    model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.3)
model.save('Models/' + STUDY_TYPE + '.model')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)
plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Losses')
plt.plot(epochs_range, val_loss, label='Validation Losses')
plt.legend(loc='upper right')
plt.title('Losses')
plt.show()
