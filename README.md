here is the link for the github repo:- 
## American Sign Language (ASL) 
### Introduction to American Sign Language 
American Sign Language (ASL) is the primary language used by many deaf individuals in North America. It is also used by hard-of-hearing and hearing individuals. ASL is as rich as spoken languages, employing signs made with the hands, along with facial gestures and bodily postures.

### Project Overview
Recent progress has been made in developing computer vision systems that translate sign language to spoken language. These systems often rely on complex neural network architectures to detect subtle patterns in streaming video. As a first step towards understanding how to build a translation system, we can reduce the problem size by translating individual letters instead of sentences.

### Project Goals
In this notebook, we will train a convolutional neural network (CNN) to classify images of ASL letters. After loading, examining, and preprocessing the data, we will train the network and test its performance.

### Data Loading
We begin by loading the training and test data.


```python
#Import packages and set numpy random seed
import numpy as np
np.random.seed(5) 
import tensorflow as tf
tf.set_random_seed(2)
from datasets import sign_language
import matplotlib.pyplot as plt
%matplotlib inline


# Load pre-shuffled training and test datasets

(x_train, y_train), (x_test, y_test) = sign_language.load_data()
```

## 2. Visualize the Training Data
Next, we create a list of string-valued labels containing the letters that appear in the dataset. We visualize the first several images in the training data along with their corresponding labels.
```python
# Store labels of dataset
labels = ['A', 'B', 'C']

# Print the first several training images, along with the labels

fig = plt.figure(figsize=(20,5))
for i in range(36):
  ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
  ax.imshow(np.squeeze(x_train[i]))
  ax.set_title("{}".format(labels[y_train[i]]))
plt.show()
```

## 3. Examine the Dataset
We examine how many images of each letter are in the dataset.
```python
# Number of A's in the training dataset
num_A_train = sum(y_train == 0)
# Number of B's in the training dataset
num_B_train = sum(y_train == 1)
# Number of C's in the training dataset
num_C_train = sum(y_train == 2)

# Number of A's in the test dataset
num_A_test = sum(y_test == 0)
# Number of B's in the test dataset
num_B_test = sum(y_test == 1)
# Number of C's in the test dataset
num_C_test = sum(y_test == 2)


# Print statistics about the dataset

print("Training set:")
print("\tA: {}, B: {}, C: {}".format(num_A_train, num_B_train, num_C_train))
print("Test set:")
print("\tA: {}, B: {}, C: {}".format(num_A_test, num_B_test, num_C_test))
```

## 4. One-hot Encode the Data
Keras models do not accept categorical integer labels. We need to one-hot encode the labels before supplying them to the model.

```python
from keras.utils import np_utils

# One-hot encode the training labels
y_train_OH = np_utils.to_categorical(y_train, 3)

# One-hot encode the test labels
y_test_OH = np_utils.to_categorical(y_test, 3)
```

## 5. Define the Model
We define a CNN to classify the data. The network accepts an image of an ASL letter as input and returns the predicted probabilities that the image belongs in each category.
```python
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Sequential

model = Sequential()
# First convolutional layer accepts image input
model.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu', 
            input_shape=(50, 50, 3)))
# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(4, 4)))
# Add a convolutional layer
model.add(Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'))
# Add another max pooling layer
model.add(MaxPooling2D(pool_size=(4, 4)))
# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Summarize the model
model.summary()

```

## 6. Compile the Model
We compile the model by specifying the optimizer, loss function, and a metric to track during training.

```python
# Compile the model

model.compile(optimizer='rmsprop', 
       loss='categorical_crossentropy', 
       metrics=['accuracy'])
```
## 7. Train the Model
We fit the model to the training data.
```python
# Train the model
hist = model.fit(x_train, y_train_OH, validation_split=0.2, epochs=2, batch_size=32)
```
## 8. Test the Model
We evaluate the model on the test dataset to determine its accuracy on unseen data.
```python
# Obtain accuracy on test set
score = model.evaluate(x_test, y_test_OH, verbose=0)
print('Test accuracy:', score[1])
```
## 9. Visualize Mistakes
Finally, we visualize the images that were incorrectly classified by the model.
```python
# Get predicted probabilities for test dataset
y_probs = model.predict(x_test)

# Get predicted labels for test dataset
y_preds = np.argmax(y_probs, axis=1)

# Indices corresponding to test images which were mislabeled
bad_test_idxs = np.where(y_preds != y_test)[0]

# Print mislabeled examples
fig = plt.figure(figsize=(25,4))
for i, idx in enumerate(bad_test_idxs):
  ax = fig.add_subplot(2, np.ceil(len(bad_test_idxs)/2), i + 1, xticks=[], yticks=[])
  ax.imshow(np.squeeze(x_test[idx]))
  ax.set_title("{} (pred: {})".format(labels[y_test[idx]], labels[y_preds[idx]]))
plt.show()
```



## i've added some more files, so let's se which files contains code for what purpose.
----
#### collectdata.py:- we've used this file to collect data and then store it inside image folder in subfiles individually.
[Click here to go directly to collectdata.py](collectdata.py)
----
#### function.py:- in this file i've included functions which we are going to use later in this project. functions to train the model, to take the output, extract keypoints from out data and form an array so that we can train our model.
[Click here to go directly to function.py](function.py)
-----
#### data.py:- this includes code which will extract the keypoints from our data, i.e. the images we've captured and store it in .mp format and store it inside MP_Data folder.
[Click here to go directly to data.py](data.py)
----
#### trainmodel.py :- in this i've built a model and trained it with the data i've taken previously.
[Click here to go directly to trainmodel.py](trainmodel.py)
----
#### app.py:- the output that we need from our model is shown here.
[Click here to go directly to app.py](app.py)
