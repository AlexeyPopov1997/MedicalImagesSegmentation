import time
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

EPOCHS = 10
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128, 256]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for layer in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(X, y, batch_size=32, epochs=EPOCHS, validation_split=0.3)

            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(10)
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
            plt.savefig('Optimizing/' + STUDY_TYPE + '/' + NAME + '.png')
            plt.show()
