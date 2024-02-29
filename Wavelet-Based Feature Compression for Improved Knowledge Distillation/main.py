#!/usr/bin/env python
# coding: utf-8

# # Histopathologic Cancer Detection
# 
# This is my entry for the Kaggle playground competition on cancer detection. This is my second machine learning project and was motivated by my completion of Course 4 of the Deeplearning.ai specialisation on Coursera. Pandas, matplotlib, experimenting with different hyper-parameters and transfer learning are among the skills I have practiced during my time doing this.
# 
# 
# 

# In[22]:


import numpy as np
import pandas as pd
import cv2
import os
from glob import glob 
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score
from scipy.special import rel_entr
from keras_preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPool2D
from scipy.special import softmax
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
from keras import optimizers
from IPython.display import clear_output

from classification_models.keras import Classifiers
ResNet18, preprocess_input = Classifiers.get('resnet18')
ResNext50, preprocess_input = Classifiers.get('resnext50')
ResNet101, preprocess_input = Classifiers.get('resnet101')
ResNet34, preprocess_input = Classifiers.get('resnet34')
ResNext101, preprocess_input = Classifiers.get('resnext101')


# ## Processing the data

# Now it's time to import the data.
# What I want to do is:
# 
# 1. Load and display some positive and negative test examples
# 2. Split the train data into train and dev sets

# In[2]:


######################################################################################################
# python kdml_classify.py -a 0.1 -b 0.2 -g 0.1 -p 4 -e 200 -n 1
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--alpha", help='alpha value', type=float, required=True)
parser.add_argument("-p", "--gpu", default='0', help='gpu number', type=str, required=True)
parser.add_argument("-e", "--epochs", type=int, required=True)
parser.add_argument("-n", "--foldernumber", type=int, required=True)
parser.add_argument("-r", "--run", type=int, required=True)
parser.add_argument("-v", "--version", type=str, required=True)
parser.add_argument("-w", "--run_wavelet", type=int, required=True, default=None)
parser.add_argument("-k", "--keep", type=int, required=True, default=None)

args = parser.parse_args()
keep = args.keep
n_classes=4
IMG_SIZE=224
BATCH_SIZE=8
GPU_NUMBER = args.gpu
num_epochs= args.epochs
os.environ['CUDA_VISIBLE_DEVICES']=GPU_NUMBER
run_wavelet=args.run_wavelet
saving_checkpoint = False
save_checkpoint_every = 20
names_model=args.version
run_model=args.run
names_model=args.version
run_model=args.run
if names_model == 'V2':
    network = "resnet50_mobilenetv2"
else :
     network = "resnet50_resnet18"

SAVE_PLOTS=True


print('Total epochs:', num_epochs)
print(f'Checkpoint Save every:{save_checkpoint_every} epochs')


# %%
if run_wavelet!=1:
    main_folder=f'{network}/single_saved_model/{names_model}_alpha_{str(args.alpha)}_run_{str(args.foldernumber)}'
    folder_teacher = f'{network}/single_saved_model/{names_model}_alpha_{str(args.alpha)}_run_{str(args.foldernumber)}/teacher'  
    folder_student1 = f'{network}/single_saved_model/{names_model}_alpha_{str(args.alpha)}_run_{str(args.foldernumber)}/student1' 
  
    print(f'Saving plots, trained_mode and losses to folder: {folder_teacher}, {folder_student1}')
else:
    main_folder=f'{network}/wavelet_saved_model/{names_model}_alpha_{str(args.alpha)}_run_{str(args.foldernumber)}_keep:{keep}'
    folder_teacher = f'{network}/wavelet_saved_model/{names_model}_alpha_{str(args.alpha)}_run_{str(args.foldernumber)}_keep:{keep}/teacher'  
    folder_student1 = f'{network}/wavelet_saved_model/{names_model}_alpha_{str(args.alpha)}_run_{str(args.foldernumber)}_keep:{keep}/student1' 

    print(f'Saving plots, trained_mode and losses to folder: {folder_teacher}, {folder_student1}')

checkpoint_folder_teacher = os.path.join(folder_teacher, 'checkpoints_teacher')
checkpoint_folder_student1 = os.path.join(folder_student1, 'checkpoints_student1')

print(f'Saving checkpoints: {checkpoint_folder_teacher}')
print(f'Saving checkpoints: {checkpoint_folder_student1}')


def check_folder(folder):
  if not os.path.exists(folder):
    os.makedirs(folder)


check_folder(folder_teacher)
check_folder(folder_student1)

loss_curves = f'classification_loss_curves_full_dataset.jpg' 
print(f'Loss Curves Path: {loss_curves}')




##############################################################################################################
data_dir="/home/usma/chaoyang-data/"

train = pd.read_csv(data_dir+'train_labels.csv')
train['label'] = train['label'].astype(str)
val = pd.read_csv(data_dir+'val_labels.csv')
val['label'] = val['label'].astype(str)
test=pd.read_csv(data_dir+'test_labels.csv')
test['label'] = test['label'].astype(str)
print(f"Train: {train.shape} \nVal: {val.shape} \nTest: {test.shape}")


train_datagen = ImageDataGenerator(rescale=1./255,
                                  vertical_flip = True,
                                  horizontal_flip = True,
                                  rotation_range=20,
                                  zoom_range=0.2, 
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.05,
                                  channel_shift_range=0.1
                                )

test_datagen = ImageDataGenerator(rescale = 1./255) 


# In[15]:



train_generator = train_datagen.flow_from_dataframe(dataframe = train, 
                                                    directory = None,
                                                    x_col = 'path', 
                                                    y_col = 'label',
                                                    target_size = (IMG_SIZE,IMG_SIZE),
                                                    class_mode = "categorical",
                                                    batch_size=BATCH_SIZE,
                                                    seed = 110318,
                                                    shuffle = True)


# In[16]:


valid_generator = test_datagen.flow_from_dataframe(dataframe = val,
                                                   directory = None,
                                                   x_col = 'path',
                                                   y_col = 'label',
                                                   target_size = (IMG_SIZE,IMG_SIZE),
                                                   class_mode = 'categorical',
                                                   batch_size = BATCH_SIZE,
                                                   shuffle = False)

test_generator = test_datagen.flow_from_dataframe(dataframe = test,
                                                   directory = None,
                                                   x_col = 'path',
                                                   y_col = 'label',
                                                   target_size = (IMG_SIZE,IMG_SIZE),
                                                   class_mode = 'categorical',
                                                   batch_size = BATCH_SIZE,
                                                   shuffle = False)



# ## Creating the model

# In[17]:




dropout_fc = 0.5
IMG_SIZE = 224
inputs = tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3)) #Resnet50
conv_base=tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet',input_tensor=inputs)

if names_model=='V1':
    x=tf.keras.layers.Conv2D(512, (1,1),  padding= 'same',name='conv5_block4_1_conv')(conv_base.output)
else:
    x=tf.keras.layers.Conv2D(1280, (1,1),  padding= 'same',name='conv5_block4_1_conv')(conv_base.output)
x=tf.keras.layers.BatchNormalization(name='conv5_block4_1_bn')(x)
x=tf.keras.layers.ReLU(name='conv5_block4_1_relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dense(n_classes, activation="softmax", name="classification")(x)
teacher = tf.keras.Model(conv_base.input,  x)


dropout_fc = 0.5
inputs = tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
if names_model == 'V1':
    base=ResNet18(input_shape=(224, 224, 3), weights='imagenet', include_top=False,input_tensor=inputs)
else:
    base=tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False,input_tensor=inputs)

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dense(n_classes, activation="softmax", name="classification")(x)
student = tf.keras.Model(base.input,  x)



def delete_old_files(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def get_best_model(folder_name):
    files=os.listdir(folder_name)
    matching = [s for s in files if 'index' in s]
    #print(matching)
    final_matching = [s[:-6] for s in matching]
    # print(final_matching)
    split_matching = [int(s.split(':')[1]) for s in final_matching]
    
    
    split_matching.sort()
    index=(split_matching[-1])
    final_file= [s for s in final_matching if str(index) in s]
    return final_file
#saved_model/V1_alpha_0.2_beta_0.2_run_1/student1





################################################################################################################################
global_train_accuracy=[]
global_val_accuracy=[]
global_train_accuracy_s1=[]
global_val_accuracy_s1=[]

global_train_loss=[]
global_val_loss=[]
global_train_loss_s1=[]
global_val_loss_s1=[]
class CustomEarlyStopping(keras.callbacks.Callback):

    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        # self.best_weights = None
        self.best_weights_teacher = None
        self.best_weights_s1 = None
        



    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best1 = 0
       
        self.best=0
        self.best_epoch_s1=0
      
        self.best_epoch=0

    def on_epoch_end(self, epoch, logs=None):
        teacher_acc= logs.get("accuracy_teacher")
        global_train_accuracy.append(np.round(teacher_acc,4))
        teacher_val_acc= logs.get("val_accuracy_teacher")
        global_val_accuracy.append(np.round(teacher_val_acc,4))
        train_current1 = logs.get("accuracy_student")
        global_train_accuracy_s1.append(np.round(train_current1,4))
        current1 = logs.get("val_accuracy_student")
        global_val_accuracy_s1.append(np.round(current1,4))
       
        
        
        teacher_loss= logs.get("teacher_loss")
        global_train_loss.append(np.round(teacher_loss,4))
        teacher_val_loss= logs.get("val_teacher_loss")
        global_val_loss.append(np.round(teacher_val_loss,4))
        train_s1_loss = logs.get("student_loss")
        global_train_loss_s1.append(np.round(train_s1_loss,4))
        s1_loss = logs.get("val_student_loss")
        global_val_loss_s1.append(np.round(s1_loss,4))
  
     
        if epoch%1==0:
          plt.figure(figsize=(20, 6))
          plt.subplot(2,4 , 1)
          plt.plot(global_train_accuracy, color="r", label="Teacher")
          plt.plot(global_train_accuracy_s1, color="b", label="S1")
       
          plt.xlabel('Epoch')
          plt.ylabel('Accuracy')
          plt.title(f'Training accuracy')
          # plt.title('Accuracy')
          plt.legend()

          plt.subplot(2, 4, 2)
          max_acc=np.max(np.array(global_val_accuracy))
          step=np.argmax(np.array(global_val_accuracy))
          
          max_acc_s1=np.max(np.array(global_val_accuracy_s1))
          step_s1=np.argmax(np.array(global_val_accuracy_s1))

          
          plt.plot(global_val_accuracy, color="r", label="Teacher")
          plt.plot(global_val_accuracy_s1, color="b", label="S1")
     
          plt.xlabel('Epoch')
          plt.ylabel('Accuracy')
          plt.title(f'Validation accuracy:- \n best_T: {max_acc} @ {step}; \n best_s1:{max_acc_s1} @ {step_s1};')
          plt.legend()
          
          
          
          plt.subplot(2, 4, 3)
          plt.plot(global_train_loss, color="r", label="Teacher")
          plt.plot(global_train_loss_s1, color="b", label="S1")
        
          plt.xlabel('Epoch')
          plt.ylabel('Accuracy')
          plt.title(f'Training loss')
          plt.legend()
          
          plt.subplot(2, 4, 4)
          
          min_loss=np.array(global_val_loss)[step]
          min_loss_s1=np.array(global_val_loss_s1)[step_s1]
     
          plt.plot(global_val_loss, color="r", label="Teacher")
          plt.plot(global_val_loss_s1, color="b", label="S1")
       
          plt.xlabel('Epoch')
          plt.ylabel('Accuracy')
          plt.title(f'Validation loss:- \n best_T: {min_loss} @ {step}; \n best_s1:{min_loss_s1} @ {step_s1}')
          plt.legend()

          if SAVE_PLOTS:
              plt.savefig(f'{main_folder}/{loss_curves}', dpi=300, bbox_inches='tight')
          else:
              plt.show()
          plt.close()
        
        # print(current1)
        flag=0
        if np.greater(current1, self.best1):
            self.wait = 0
        if np.greater(teacher_val_acc, self.best):
            self.best= teacher_val_acc
            self.best_epoch=epoch
            self.best_weights_teacher = self.model.teacher.get_weights()
            delete_old_files(folder_teacher)
            print(f'\n Teacher Model save at {self.best_epoch}\n')
            self.model.teacher.save_weights(folder_teacher+"/teacher@:"+str(self.best_epoch))
        else:
            self.wait += 1
            flag=1

        if np.greater(current1, self.best1):
            print(f'\n value of S1  improved from  {np.round(self.best1,4)} to {np.round(current1,4)}')
            self.best1 = current1
            self.best_epoch_s1=epoch
            # Record the best weights if current results is better (less).
            # self.best_weights = self.model.get_weights()
            self.best_weights_s1 = self.model.student1.get_weights()
            # print(self.model.student1)
            delete_old_files(folder_student1)
            print(f'\n S1 Model save at {self.best_epoch_s1}\n')
            self.model.student1.save_weights(folder_student1+"/student1@:"+str(self.best_epoch_s1))
            
            # print(f'value of S1 IOU improved from  {self.best1} and S2 IOU {self.best2}')
        
        else:
            if flag==0:
                self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                # self.model.set_weights(self.best_weights)
                self.model.teacher.set_weights(self.best_weights_teacher)
                self.model.student1.set_weights(self.best_weights_s1)
              
            else:
                print(f'value of S1 not improved  {np.round(self.best1,4)}')
                

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

################################################################################################################################

import pywt
wavelet = 'haar'

level = 2
def get_wavelets(data,s=0,p=1):

  # Perform wavelet transform
 
  coeffs = pywt.wavedec(data, wavelet, level=level)
  coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)
  coeff_magnitudes_sorted = np.sort(np.abs(coeffs_arr.reshape(-1)))
  if p!=1:
    reconstr = pywt.waverec(coeffs, wavelet)
    # print(f"*********************{tf.reduce_all(tf.equal(data, reconstr))}************************************")
    return reconstr
  if s == 1:

    thresholds = coeff_magnitudes_sorted[int(np.floor(len(coeff_magnitudes_sorted) * (1 - keep/ 100)))]
    ind = np.abs(coeffs_arr)>thresholds
    cfilt = coeffs_arr * ind
    coeffs_filt=pywt.array_to_coeffs(cfilt,coeffs_slices,output_format='wavedec')
    recon = pywt.waverec(coeffs_filt, wavelet)
    return recon
  return coeff_magnitudes_sorted



teacher_metric = keras.metrics.CategoricalAccuracy()
student1_metric= keras.metrics.CategoricalAccuracy()

class Distiller(keras.Model):
    def __init__(self, student1,teacher,student1_layer,teacher_layer,wavelets):
        super(Distiller, self).__init__()
        self.student1 = student1
        
        self.teacher = teacher
        self.epoch_flag=0
        self.wavelets=wavelets
        self.student1_layer=student1_layer
        self.teacher_layer=teacher_layer
        #self.num_classes=10
    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        distillation_loss_fn1,
        alpha=0.1,
        temperature=3,
        ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics,run_eagerly=True)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.distillation_loss_fn1 = distillation_loss_fn1
        self.alpha = alpha
   
        self.temperature = temperature
    
    def call(self, input):
      dec1_output = self.teacher(input)
      # print("1",dec1_output.shape)
      dec2_output = self.student1(input)
      
      # print("2",dec2_output.shape)
      return dec1_output, dec2_output

      
    def train_step(self, data):
        # Unpack data
        x, y = data


        # Forward pass of teacher

        out_list=[]
        teacher_grad = tf.keras.models.Model([self.teacher.inputs],
                                   [self.teacher.get_layer(self.teacher_layer).output,
                                    self.teacher.output])
     
        student1_grad_model = tf.keras.models.Model([self.student1.inputs],
                                          [self.student1.get_layer(self.student1_layer).output,
                                            self.student1.output])
       
        with tf.GradientTape(persistent=True) as tape:

            # Forward pass of student
            teacher_predictions = self.teacher(x, training=True)
            teacher_conv_outputs, teacher_preds = teacher_grad(x)
            
            
            # print(teacher_predictions.shape)
            student1_predictions = self.student1(x, training=True)
            student1_conv_outputs, student1_preds = student1_grad_model(x)
           
            teacher_wavelets1 = get_wavelets(teacher_predictions.numpy().flatten(),p=1)
            student1_wavelets1 = get_wavelets(student1_predictions.numpy().flatten(),p=1)
            student1_wavelets2 = get_wavelets(student1_conv_outputs.numpy().flatten(),s=1)
            student1_wavelets2 = student1_wavelets2.reshape(student1_conv_outputs.shape)
            teacher_wavelets2 = get_wavelets(teacher_conv_outputs.numpy().flatten(),s=1)
            teacher_wavelets2 = teacher_wavelets2.reshape(teacher_conv_outputs.shape)

           
            # Compute losses
            teacher_loss=self.student_loss_fn(y,teacher_predictions)
            student1_loss = self.student_loss_fn(y, student1_predictions)
            
            # if student1_loss > student2_loss:
            #    print("student 1 loss",student1_loss)

            if self.wavelets != 1:
              distillation_loss =self.distillation_loss_fn1(teacher_predictions, student1_predictions)
              loss = self.alpha * student1_loss + ((1 - self.alpha)) * distillation_loss
     
            else:
               distillation_loss= (
                  self.distillation_loss_fn1(
                      (teacher_predictions),
                      (student1_predictions ),
                  )+
                  
                  self.distillation_loss_fn(
                      (teacher_wavelets2),
                      (student1_wavelets2),
                  )
                  
              )
               loss = self.alpha * student1_loss +((1 - self.alpha)) * (distillation_loss)

            

            
        # Compute gradients
        # if student1_loss>student2_loss:
        # tf.print(student1_loss)
        trainable_vars1 = self.student1.trainable_variables
        gradients1 = tape.gradient(loss, trainable_vars1)
        # else:
        
        teacher_var= self.teacher.trainable_variables
        gradients = tape.gradient(teacher_loss, teacher_var)
        # Update weights
        # if student1_loss>student2_loss:
        self.optimizer.apply_gradients(zip(gradients1, trainable_vars1))
        # else:
        
        self.optimizer.apply_gradients(zip(gradients,teacher_var ))
        # Update the metrics configured in `compile()`.
        # self.compiled_metrics.update_state(y, student1_predictions)
        teacher_metric.update_state(y,teacher_predictions)
        student1_metric.update_state(y,student1_predictions)
       
        # Return a dict of performance
        # results = {"Accuracy for student 1": m.result() for m in self.metrics}
        results={"accuracy_teacher": teacher_metric.result(), 
                 "accuracy_student": student1_metric.result()}

        results.update(
            {"teacher_loss": teacher_loss,"student_loss": student1_loss, "distillation_loss_S1": distillation_loss}
        )
        '''results.update(
            {"Teacher_loss": teacher_loss,"student1_loss": loss1}
        )'''
        
        
        return results
    @property
    def metrics(self):

        return [teacher_metric, student1_metric]
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_teacher = self.teacher(x,training=False)
        y_prediction1 = self.student1(x, training=False)
        
        # Calculate the loss
        teacher_loss = self.student_loss_fn(y,y_teacher)
        student1_loss = self.student_loss_fn(y, y_prediction1)
       
        # Update the metrics.
        teacher_metric.update_state(y,y_teacher)
        student1_metric.update_state(y,y_prediction1)
        
        # Return a dict of performance
        # results = {m.name: m.result() for m in self.metrics}
        results={"accuracy_teacher":teacher_metric.result(),"accuracy_student": student1_metric.result()}
        results.update({"teacher_loss":teacher_loss,"student_loss": student1_loss,})
        return results


# In[ ]:


if names_model == 'V1':
    distiller = Distiller(student1=student,teacher=teacher,
                            student1_layer='stage4_unit2_conv2',teacher_layer='conv5_block3_2_conv',wavelets=run_wavelet) #Resnet50-resnet18
else:
    distiller = Distiller(student1=student,teacher=teacher,
                          student1_layer='Conv_1',teacher_layer='conv5_block4_1_conv',wavelets=run_wavelet) #resnet50-mobilenet
# distiller = Distiller(student1=student,teacher=teacher,
#                           student1_layer='conv_pw_13',teacher_layer='conv5_block4_1_conv',wavelets=run_wavelet) #resnet50-mobilenet

distiller.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        # optimizer=tf.keras.optimizers.SGD(
        # learning_rate=0.001, momentum=0.9, nesterov=False, name="SGD"),
    metrics=['accuracy'],
    student_loss_fn=keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=keras.losses.MeanSquaredError(),
    distillation_loss_fn1=keras.losses.KLDivergence(),
    alpha=args.alpha,
    temperature=2,
)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy_student', patience=10, verbose=1, factor=0.5, min_lr=1e-7)
checkpoint = CustomEarlyStopping(patience=40)
train_step_size = train_generator.n // train_generator.batch_size
valid_step_size = valid_generator.n // valid_generator.batch_size

if run_model==1:
    history1 = distiller.fit(train_generator,
                                        steps_per_epoch = train_step_size,
                                        epochs = num_epochs,
                                        validation_data = valid_generator,
                                        validation_steps = valid_step_size,
                                        callbacks =  [checkpoint,learning_rate_reduction],
    #                                      verbose = 2
                            )
elif run_model==-1:
    t_model=get_best_model(folder_teacher)
    s1_model=get_best_model(folder_student1)
    history1 = distiller.fit(train_generator,
                                        steps_per_epoch = train_step_size,
                                        epochs = num_epochs,
                                        validation_data = valid_generator,
                                        validation_steps = valid_step_size,
                                        callbacks =  [checkpoint],
    #                                      verbose = 2
                            )
else:
    t_model=get_best_model(folder_teacher)
    s1_model=get_best_model(folder_student1)

    path_t=folder_teacher+'/'+t_model[0]

    path_s1=folder_student1+'/'+s1_model[0]


    print("Teacher Model loaded",path_t)

    print("S1 Model loaded",path_s1)

    teacher.load_weights(path_t)
    student.load_weights(path_s1)





# In[89]:


from sklearn.metrics import accuracy_score
print("--------------------------------------------------- valid ----------------------------")

distiller.evaluate(valid_generator)
data=[]
testLabels=[]
for i in range(len(valid_generator)):
    data.extend(valid_generator[i][0])
    testLabels.extend(valid_generator[i][1])
data=np.array(data)
testLabels=np.array(testLabels)


t=teacher.predict(data)

t2,s1=distiller.predict(data)
s11=student.predict(data)

auc = roc_auc_score(testLabels, t)
auc1 = roc_auc_score(testLabels, s1)


t = (t == t.max(axis=1)[:, None]).astype(int)

s1 = (s1 == s1.max(axis=1)[:, None]).astype(int)

s11 = (s11 == s11.max(axis=1)[:, None]).astype(int)

t2 = (t2 == t2.max(axis=1)[:, None]).astype(int)

# Calculate precision
precision = precision_score(testLabels, t, average='macro')
# Calcula, average='macroe recall
recall = recall_score(testLabels, t,  average='macro')
# Calculate F1-score
f1 = f1_score(testLabels, t,  average='macro')
print("VAL Teacher acc",accuracy_score(testLabels, t))
print("Teacher Precision:", precision)
print("Teacher Recall:", recall)
print("Teacher F1-Score:", f1)
print('Teacher ROC AUC:', auc)

print("\n\n")

precision = precision_score(testLabels, s11, average='macro')
# Calculate recall
recall = recall_score(testLabels, s11, average='macro')
# Calculate F1-score
f1 = f1_score(testLabels, s11,  average='macro')

print("Student Valid acc",accuracy_score(testLabels, s11))
print("Student Precision:", precision)
print("Student Recall:", recall)
print("Student F1-Score:", f1)
print('Student 1 ROC AUC:',auc1)


print("--------------------------------------------------- test ----------------------------")
distiller.evaluate(test_generator)

data=[]
testLabels=[]
for i in range(len(test_generator)):
    data.extend(test_generator[i][0])
    testLabels.extend(test_generator[i][1])
data=np.array(data)
testLabels=np.array(testLabels)


t=teacher.predict(data)
t2,s1=distiller.predict(data)
s11=student.predict(data)



print("##########################################################")

KL_s1= np.mean(tf.keras.losses.kullback_leibler_divergence(
                t,
                s11,
            ))


print("Student 1 fidelity KL",KL_s1)

print("##########################################################")

auc = roc_auc_score(testLabels, t)
auc1 = roc_auc_score(testLabels, s1)


t = (t == t.max(axis=1)[:, None]).astype(int)

s1 = (s1 == s1.max(axis=1)[:, None]).astype(int)

s11 = (s11 == s11.max(axis=1)[:, None]).astype(int)

t2 = (t2 == t2.max(axis=1)[:, None]).astype(int)

# Calculate precision
precision = precision_score(testLabels, t,  average='macro')
# Calculate recall
recall = recall_score(testLabels, t,  average='macro')
# Calculate F1-score
f1 = f1_score(testLabels, t,  average='macro')
print("Test Teacher acc",accuracy_score(testLabels, t))
print("Teacher Precision:", precision)
print("Teacher Recall:", recall)
print("Teacher F1-Score:", f1)
print('Teacher ROC AUC:', auc)

print("\n\n")

precision = precision_score(testLabels, s11, average='macro')
# Calculate recall
recall = recall_score(testLabels, s11,  average='macro')
# Calculate F1-score
f1 = f1_score(testLabels, s11,  average='macro')

print("Student Test acc",accuracy_score(testLabels, s11))
print("Student Precision:", precision)
print("Student Recall:", recall)
print("Student F1-Score:", f1)
print('Student 1 ROC AUC:',auc1)


print("###############################################################################################")

print(f'ONLINE PREDICTIONS CLASSIFICATION KD {network} SHARING only ALPHA:{args.alpha} & MODEL:{names_model} & RUN:{str(args.foldernumber)}, wavelet:{run_wavelet}_keep:{keep}')
