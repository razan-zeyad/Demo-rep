import os
import numpy as np
import pandas as pd
import pickle
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.applications import EfficientNetV2L, EfficientNetB5
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

def lr_scheduler(epoch, lr):
    if epoch < 40:
        return lr
    else:
         return lr *0.8
def process_data(df, img_size):
    images = []

    for _, row in df.iterrows():
        #print(row['image'])
        img = cv2.imread(row['image'])
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
    
    images = np.array(images)
    images = images.astype('float32') / 255.0  # normalization
    #print(images.shape)
    y=df['Labels'].values
    #print(y[:5])
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    #specify the classes sequence before encoding
    label_encoder.classes_=np.array(['low', 'medium','high'])
    # Transform your labels using the fitted encoder
    y = label_encoder.transform(y)

    #print (y[:5])

    y=y.reshape(-1,1)
    #print(y[:5])

    ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
    labels_encoded = ct.fit_transform(y) #.toarray()
    """for img in images[:5]:
        plt.imshow(img)
        plt.show()"""
    #print(labels_encoded[:5])
   
    images, labels_encoded = shuffle(images, labels_encoded, random_state=1)
    """for img in images[:5]:
        plt.imshow(img)
        plt.show()"""
    #print(labels_encoded[:5])
    return images, labels_encoded



os.chdir(r'D:\razan\GP')
img_size = 480
epochs = list(range(1, 81))


dataset_folders = os.listdir('TrainData')

train_images_labeled = [] # will contain all the image labels and directories
test_images_labeled = []
for folder in dataset_folders:
    train_images = os.listdir('TrainData' + '/' +folder)
    test_images = os.listdir('TestData' + '/' +folder)
    
 # Added images to a list 
    for image in train_images:
        train_images_labeled.append((folder, str('TrainData' + '/' +folder) + '/' + image))
    for image in test_images:
        test_images_labeled.append((folder, str('TestData' + '/' +folder) + '/' + image))
        


df = pd.DataFrame(data=train_images_labeled, columns=['Labels', 'image'])

df2 = pd.DataFrame(data=test_images_labeled, columns=['Labels', 'image'])
#print(df["Labels"].value_counts())

#print("Total number of images in the test: ", len(df2))

# Process training data

train_images, train_labels_encoded = process_data(df, img_size)

# Process test/validation data
test_images, test_labels_encoded = process_data(df2, img_size)

test_x, val_x, test_y, val_y = train_test_split(test_images, test_labels_encoded,
                                                test_size=0.2, random_state=415)



#inspect the shape of the training and testing.
#print(train_images.shape)
#print(train_labels_encoded.shape)
#print(test_x.shape)
#print(test_y.shape)
#print(val_x.shape)
#print(val_y.shape)

PTModel = EfficientNetV2L(weights='imagenet', include_top=False,classes=3, 
                          input_shape=(img_size, img_size, 3))
for layer in PTModel.layers:
    layer.trainable = False

x = Flatten()(PTModel.output)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=PTModel.input, outputs=predictions)
# Create a learning rate scheduler callback
lr_callback = LearningRateScheduler(lr_scheduler)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])



#model.summary()

hist = model.fit(train_images, train_labels_encoded, batch_size=32,epochs=80, 
                 validation_data=(val_x, val_y), verbose=2,callbacks=[lr_callback])
test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=2)
print("Test Accuracy:", test_accuracy)

hist={'loss': hist.history['loss'], 'val_loss': hist.history['val_loss'],
        'accuracy': hist.history['accuracy'], 'val_accuracy': hist.history['val_accuracy']}
# Save the training history to a file
with open('training_history.pkl', 'wb') as f:
    pickle.dump(hist, f)


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, hist['loss'],label='Training Loss')
plt.plot(epochs, hist['val_loss'], label='Validation Loss')
plt.title('Model Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True) 

plt.subplot(1, 2, 2)
plt.plot(epochs, hist['accuracy'], label='Training Accuracy')
plt.plot(epochs, hist['val_accuracy'], label='Validation Accuracy')
plt.title('Model Training and Validation Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True) 
plt.tight_layout()
plt.show()

###############
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Make predictions on the test set
predictions = model.predict(test_x)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_y, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['low', 'medium', 'high'], yticklabels=['low', 'medium', 'high'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
#############