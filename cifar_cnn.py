import tensorflow as tf
from tensorflow.keras import datasets, layers, models 
import matplotlib.pyplot as plt

(training_data, train_labels), (test_data, test_labels) = datasets.cifar10.load_data()

training_data = training_data/255.0
test_data = test_data/255.0

model = models.Sequential()
model.add(layers.Conv2D(filters=24,kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2))
model.add(layers.Conv2D(filters=48,kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
              metrics=['accuracy'])
results = model.fit(training_data, train_labels, epochs=10, validation_data=(test_data,test_labels))
plt.plot(results.history['accuracy'], label='accuracy')
plt.plot(results.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')