# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image

# #  Model deployment
# In[2]:
get_ipython().system(' pip install tensorflow')
get_ipython().system(' pip install keras')
#Defining paths
TRAIN_PATH = "biodegradable"
VAL_PATH = "NON_BIODEGRADABLE"

# In[3]:
#Training model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])
# In[4]:
#Getting parameters
model.summary()
# # Training data 
# In[5]:
#Moulding train images
train_datagen = image.ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)
test_dataset = image.ImageDataGenerator(rescale=1./255)
# In[6]:
#Reshaping test and validation images 
train_generator = train_datagen.flow_from_directory(
    'NON_BIODEGRADABLE/Train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')
validation_generator = test_dataset.flow_from_directory(
    'NON_BIODEGRADABLE/Val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')
# In[7]:
224/32
# In[8]:
#Training the model
hist_new = model.fit_generator(
    train_generator,
    steps_per_epoch=7,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps=2
)
# In[10]:
#Getting summary
summary=hist_new.history
print(summary)
# In[11]:
model.save("model_covid.h5")


# In[13]:


model.evaluate_generator(train_generator)


# In[14]:


print(model.evaluate_generator(validation_generator))


# ## Confusion Matrix

# In[15]:


import os
train_generator.class_indices


# In[16]:


y_actual, y_test = [],[]


# In[17]:


for i in os.listdir("./NON_BIODEGRADABLE/Val/Normal/"):
    img=image.load_img("./NON_BIODEGRADABLE/Val/Normal/"+i,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict_classes(img)
    y_test.append(pred[0,0])
    y_actual.append(1)
    


# In[18]:


for i in os.listdir("./NON_BIODEGRADABLE/Val/Covid/"):
    img=image.load_img("./NON_BIODEGRADABLE/Val/Covid/"+i,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict_classes(img)
    y_test.append(pred[0,0])
    y_actual.append(0)


# In[19]:


y_actual=np.array(y_actual)
y_test=np.array(y_test)


# In[20]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
cn=confusion_matrix(y_actual,y_test)
# In[21]:
sns.heatmap(cn,cmap="plasma",annot=True) #0: Covid ; 1: Normal
