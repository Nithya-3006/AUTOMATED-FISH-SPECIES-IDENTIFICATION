import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import os.path
import cv2
from pathlib import Path
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


tf.__version__


img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                            #rotation_range=90,
                            brightness_range=(0.5,1), 
                            #shear_range=0.2, 
                            #zoom_range=0.2,
                            channel_shift_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            rescale=1./255,
                            validation_split=0.3)
                            
                            
image_dir = Path('/content/drive/MyDrive/fish/fish')


# Get filepaths and labels
# .glob() will return all file paths that match a specific pattern 
filepaths = list(image_dir.glob(r'**/*.png'))
# split a file path to head and tail (read this to understand more : https://www.geeksforgeeks.org/python-os-path-split-method/)
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))


#convert the list we made into a pandas series datatype
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')


# Concatenate filepaths and labels --> We will obtained a labeled dartaset
image_df = pd.concat([filepaths, labels], axis=1)


# Drop GT images
image_df = image_df[image_df['Label'].apply(lambda x: x[-2:] != 'GT')]


root_dir = '/content/drive/MyDrive/fish/fish'


img_generator_flow_train = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224,224),
    batch_size=16,
    shuffle=True,
    subset="training")
img_generator_flow_valid = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224,224),
    batch_size=16,
    shuffle=True,
    subset="validation")
    
    
len(image_df)
    
        
imgs, labels = next(iter(img_generator_flow_train))
for img, label in zip(imgs, labels):
    plt.imshow(img)
    plt.show()
    
image_df['Label'].value_counts(ascending=True)


base_model = tf.keras.applications.InceptionV3(input_shape=(224,224,3),
                                               include_top=False,
                                               weights = "imagenet"
                                               )


base_model.trainable = False


model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(9, activation="softmax")
])


model.summary()


# Shuffle the DataFrame and reset index
image_df = image_df.sample(frac=1).reset_index(drop = True)
# Show the result
image_df.head(3)


# Separate in train and test data
train_df, test_df = train_test_split(image_df, train_size=0.9, shuffle=True, random_state=1)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = [tf.keras.metrics.CategoricalAccuracy()])
              
                            
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
    
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
rotated_image = image_gen.flow_from_dataframe(
    dataframe=image_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    batch_size=32,
    seed=42,
)


augmented_images = [rotated_image[0][0][0] for i in range(5)]
plotImages(augmented_images)


image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
zoomed_image = image_gen.flow_from_dataframe(
    dataframe=image_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    batch_size=32,
    seed=42,
)


augmented_images = [zoomed_image[0][0][0] for i in range(5)]
plotImages(augmented_images)


image_gen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
)
train_data_gen = image_gen.flow_from_dataframe(
    dataframe=image_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    batch_size=32,
    seed=42,
)


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)


train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)
val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)


augmented_images = [train_images[0][0][0] for i in range(5)]
plotImages(augmented_images)


# Load the pretained model
#algorithm - neural network - deep learning
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False


inputs = pretrained_model.input


x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x) #relu is the activation function for neurla network task


outputs = tf.keras.layers.Dense(9, activation='softmax')(x) #softmax is the activation function for classiciation task


model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()


# Predict the label of the test_images
pred = model.predict(test_images)
pred = np.argmax(pred,axis=1)
# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]
# Display the result
print(f'The first 5 predictions: {pred[:5]}')


model.fit(img_generator_flow_train, 
          validation_data=img_generator_flow_valid, 
          steps_per_epoch=10, epochs=50)
