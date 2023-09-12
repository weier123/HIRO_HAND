import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random
import tensorboard
from sklearn.preprocessing import StandardScaler
from keras import backend as K


scaler = StandardScaler()   
#本程序的主要功能是将图片中的right left去掉
datadir = "Track"
#使用的照片数量
img_num = 175
#定义数据每一行的标签
#columns = ['center','left','right','steering','throttle','reverse','speed']
#to do
columns = ['center','sensor1','sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9','sensor10','sensor11','sensor12','sensor13','sensor14','sensor15']
#读取.csv的数据
data = pd.read_csv('C:/Users/X1 Carbon/Desktop/Tsinghua/demo/driving_log.csv',names = columns)
pd.set_option('display.expand_frame_repr', False)
print(data.head())
#split是内置函数分隔符，用来分割字符，按照空格，制表符等
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

#分别找出center left right的数据
data['center'] = data['center'].apply(path_leaf)
#data['left'] = data['left'].apply(path_leaf)
#data['right'] = data['right'].apply(path_leaf)
data.head()

num_bins = 25
samples_per_bin = 200
#绘制直方图
#hist, bins = np.histogram(data['steering'], num_bins)
#center = (bins[:-1]+ bins[1:]) * 0.5
#plt.bar(center, hist, width=0.05)
#plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))

#print('total data:', len(data))
#remove_list = []
#for j in range(num_bins):
#  list_ = []
#  for i in range(len(data['steering'])):
#    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
#      list_.append(i)
#  list_ = shuffle(list_)
#  list_ = list_[samples_per_bin:]
#  remove_list.extend(list_)
 
#print('removed:', len(remove_list))
#data.drop(data.index[remove_list], inplace=True)
#print('remaining:', len(data))
 
#hist, _ = np.histogram(data['steering'], (num_bins))
#plt.bar(center, hist, width=0.05)
#plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))


#print(data.iloc[1])
def load_img_steering(datadir, df):
  image_path = [[]for i in range(len(data))]
  steering = [[]for i in range(len(data))]
  #steering = np.zeros((len(data),3),int)
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center= indexed_data[0]
    image_path[i].append(os.path.join(datadir, center.strip()))
    steering[i].append(float(indexed_data[1]))
    # left image append
    #image_path.append(os.path.join(datadir,left.strip()))
    steering[i].append(float(indexed_data[2]))
    # right image append
    #im age_path.append(os.path.join(datadir,right.strip()))
    #steering.append(float(indexed_data[3])-0.15)
    steering[i].append(float(indexed_data[3]))
    steering[i].append(float(indexed_data[4]))
    steering[i].append(float(indexed_data[5]))
    steering[i].append(float(indexed_data[6]))
    steering[i].append(float(indexed_data[7]))
    steering[i].append(float(indexed_data[8]))
    steering[i].append(float(indexed_data[9]))
    steering[i].append(float(indexed_data[10]))
    steering[i].append(float(indexed_data[11]))
    steering[i].append(float(indexed_data[12]))
    steering[i].append(float(indexed_data[13]))
    steering[i].append(float(indexed_data[14]))
    steering[i].append(float(indexed_data[15]))

  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings
 
image_paths, steerings = load_img_steering('C:/Users/X1 Carbon/Desktop/Tsinghua/demo/IMG', data)
print(steerings)

random.seed()   
#训练集测试集的分割
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

#X_train = image_paths
#y_train = steerings
#X_valid = image_paths
#y_valid = steerings
#X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.4, random_state=6)


print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))
print(X_train)
print(X_train.ndim)
print(type(X_train))
print(y_train)
print(y_train.ndim)
print(type(y_train))
#fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
#axes[0].set_title('Training set')
#axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
#axes[1].set_title('Validation set')

def zoom(img):
    zoom = iaa.Affine(scale=(1, 1.3))
    img = zoom.augment_image(img)
    return img
#to do 这里需要修改用照片的数量
image = image_paths[random.randint(0, img_num)]
print(image)
original_image = mpimg.imread(image[0])
zoomed_image = zoom(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')
 
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image')

def pan(image):
    pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image
#需要修改照片数量
image = image_paths[random.randint(0, img_num)]
original_image = mpimg.imread(image[0])
panned_image = pan(original_image)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')
 
axs[1].imshow(panned_image)
axs[1].set_title('Panned_Image')

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

image = image_paths[random.randint(0, img_num)]
original_image = mpimg.imread(image[0])
brightness_altered_image = img_random_brightness(original_image)
 
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
 
axs[1].imshow(brightness_altered_image)
axs[1].set_title('Brightness altered image ')


def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle


random_index = random.randint(0, img_num)
image = image_paths[random_index]
steering_angle = steerings[random_index]
 
 
original_image = mpimg.imread(image[0])
flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)
 
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))
 
axs[1].imshow(flipped_image)
axs[1].set_title('Flipped Image - ' + 'Steering Angle:' + str(flipped_steering_angle))
def random_augment(image, steering_angle):
    image = mpimg.imread(image[0])
    #if np.random.rand() < 0.5:
    #  image = pan(image)
    #if np.random.rand() < 0.5:
    #  image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    #if np.random.rand() < 0.5:#反转不可以
    #  image, steering_angle = img_random_flip(image, steering_angle)
    
    return image, steering_angle

ncol = 2
nrow = 10
 
fig, axs = plt.subplots(nrow, ncol, figsize=(15, 50))
fig.tight_layout()
 
for i in range(10):
  randnum = random.randint(0, len(image_paths) - 1)
  random_image = image_paths[randnum]
  random_steering = steerings[randnum]
    
  original_image = mpimg.imread(random_image[0])
  augmented_image, steering = random_augment(random_image, random_steering)
    
  axs[i][0].imshow(original_image)
  axs[i][0].set_title("Original Image")
  
  axs[i][1].imshow(augmented_image)
  axs[i][1].set_title("Augmented Image")

def img_preprocess(img):
    img = img[:,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (320, 160))
    #img = img/255
    img = tf.image.per_image_standardization(img)
    #img = img/127.5-1
    return img  

image = image_paths[img_num]
original_image = mpimg.imread(image[0])
preprocessed_image = img_preprocess(original_image)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(preprocessed_image)
axs[1].set_title('Preprocessed Image')

def batch_generator(image_paths, steering_ang, batch_size, istraining):
      
  while True:
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)
      #print(random_index)
      if istraining:
        img, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
      else:
        img = mpimg.imread(image_paths[random_index][0])
        steering = steering_ang[random_index]
      
      img = img_preprocess(img)
      batch_img.append(img)
      #print(batch_img[1])
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering)) 

#做数据归一化
#x=(x-u) /std (u是均值 std是方差)
#from sklearn.preprocessing import StandardScaler  # StandardScaler 作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。
#scaler = StandardScaler()
#x_train:[None,28,28] (三维矩阵) -->[None,784](转化为二维数据)
'''x_train_scaled=scaler.fit_transform( #对训练集做归一化  fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等
    X_train.astype(np.float32).reshape(-1,1) #astype() 修改数据类型
).reshape(-1,28,28)

#对验证集做归一化
x_valid_scaled=scaler.transform(X_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled=scaler.transform(
    X_valid.astype(np.float32).reshape(-1,1)
).reshape(-1,28,28)
'''
def normlization(x,MAX,MIN):
      x = (x-MIN)/(MAX-MIN)
      return x

MAX = 1
MIN = 0
y_train = normlization(y_train,MAX,MIN)
print("y train:")
#print(y_train)
y_valid = normlization(y_valid,MAX,MIN)
x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 50, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 40, 0))
print("y_train_gen")
#print(y_train_gen)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
 
axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training Image')
 
axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation Image')

def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24,(5,5),strides=(2, 2),input_shape=(160, 320, 3),activation='relu'))
    
    model.add(Convolution2D(36,(5,5),strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())    
    model.add(keras.layers.Activation('relu'))
    model.add(Convolution2D(48,(5,5),strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    #model.add(Convolution2D(64,(3,3)))
    #model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Activation('elu'))
    model.add(Convolution2D(64,(3,3),activation='relu'))
    #
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(50, activation = 'relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(Dropout(0.5))
  
    #model.add(Dense(50, activation = 'elu'))
    #model.add(keras.layers.BatchNormalization())

    #model.add(Dropout(0.5))

    model.add(Dense(10, activation = 'relu',name="dense_2"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.5))
 
    model.add(Dense(15,name="dense_3"))
    #model.add(keras.layers.BatchNormalization())

    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    #print(model.layers[5].output)
    return model

model = nvidia_model()
#print(model.summary())
# history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid),batch_size=100,verbose=1,shuffle=1)
''' 
history = model.fit_generator(batch_generator(X_train, y_train, 75, 1),
                                  steps_per_epoch=300, 
                                  epochs=15  ,
                                  validation_data=batch_generator(X_valid, y_valid, 50, 1),
                                  validation_steps=50,
                                  verbose=1,
                                  shuffle = 1)
'''
history = model.fit_generator(batch_generator(X_train, y_train, 75, 1),
                                  steps_per_epoch=45, 
                                  epochs=100,
                                  validation_data=batch_generator(X_valid, y_valid, 50, 1),
                                  validation_steps=20,
                                  verbose=1,
                                  shuffle = 1)


#print(model.layers[5].output)
#dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('dense_3').output)
#dense1_output = dense1_layer_model.predict(x_train_gen)
#print(dense1_output.shape)
#print(dense1_output)
#inp = model.input                                           # input placeholder
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function
#print(functor)
# Testing
#test = np.random.random(input_shape)[np.newaxis,...]
#layer_outs = functor([test, 1.])
#print layer_outs
#model_path = "C:/Users/X1 Carbon/Desktop/Tsinghua/Self-Driving-Cars-master/model.h5"
#model = tf.keras.models.load_model(model_path)
#predict = model.predict(x_train_gen)
#print(x_train_gen)
#print(predict)
plt.plot(history.history['loss'])
plt.xlim((0, 15))
print(history.history['loss'])
print(history.history['val_loss'])
plt.plot(history.history['val_loss'])
for i in range(100):
      print(history.history['loss'][i])
      print(history.history['val_loss'][i])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.figure()
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
model.save('model.h5')
#plt.show()
#from google.colab import files
#files.download('model.h5')