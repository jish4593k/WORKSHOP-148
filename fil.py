import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, Canvas, PhotoImage
from scipy.io import loadmat
from time import time

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))


model = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu', input_dim=784),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

start_time = time()
model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)


y_pred = np.argmax(model.predict(x_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy on test set: %.3f%%" % (accuracy * 100))

def display_random_image():
    idx = np.random.randint(len(x_test))
    image = x_test[idx].reshape((28, 28)) * 255.0
    label = y_test[idx]
    prediction = np.argmax(model.predict(np.expand_dims(x_test[idx], axis=0)))

    canvas.delete("all")
    img = PhotoImage(data=image.flatten().tobytes(), width=28, height=28)
    canvas.create_image(0, 0, anchor="nw", image=img)
    canvas.image = img

    label_text.set(f"True Label: {label}\nModel Prediction: {prediction}")

root = Tk()
root.title("Fashion MNIST Classifier")


canvas = Canvas(root, width=28, height=28)
canvas.pack()

label_text = StringVar()
label_text.set("True Label: -\nModel Prediction: -")
label = Label(root, textvariable=label_text)
label.pack()


button = Button(root, text="Show Random Image", command=display_random_image)
button.pack()

root.mainloop()
