# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:24:35 2021

@author: Alba Casillas Rodríguez
"""

import tensorflow as tf


#Codiguillo para probar que uso las GPU y que no explota mi ordenador.

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print()
#print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))



import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.utils import np_utils


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import EarlyStopping

# Importar el optimizador a usar
# https://stackoverflow.com/questions/67604780/unable-to-import-sgd-and-adam-from-keras-optimizers
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop

# Importar el conjunto de datos
from keras.datasets import cifar100


#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# A esta función sólo se le llama una vez. Devuelve 4 vectores conteniendo,
# por este orden, las imágenes de entrenamiento, las clases de las imágenes
# de entrenamiento, las imágenes del conjunto de test y las clases del
# conjunto de test.

def cargarImagenes():
  # Cargamos Cifar100. Cada imagen tiene tamaño (32, 32, 3).
  # Nos vamos a quedar con las imágenes de 25 de las clases.
  
  (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  
  train_idx = np.isin(y_train, np.arange(25))
  train_idx = np.reshape(train_idx,-1)
  x_train = x_train[train_idx]
  y_train = y_train[train_idx]
  
  test_idx = np.isin(y_test, np.arange(25))
  test_idx = np.reshape(test_idx, -1)
  x_test = x_test[test_idx]
  y_test = y_test[test_idx]
  
  # Transformamos los vectores de clases en matrices. Cada componente se convierte en un vector
  # de ceros con un uno en la componente correspondiente a la clase a la que pertenece la imagen.
  # Este paso es necesario para la clasificación multiclase en keras.
  y_train = np_utils.to_categorical(y_train, 25)
  y_test = np_utils.to_categorical(y_test, 25)
  
  return x_train, y_train, x_test, y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve el accuracy de un modelo, definido como el 
# porcentaje de etiquetas bien predichas frente al total de etiquetas.
# Como parámetros es necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de keras (matrices
# donde cada etiqueta ocupa una fila, con un 1 en la posición de la clase
# a la que pertenece y 0 en las demás).

def calcularAccuracy(labels, preds):
  labels = np.argmax(labels, axis = 1)
  preds = np.argmax(preds, axis = 1)
  
  accuracy = sum(labels == preds)/len(labels)
  
  return accuracy

# Funcion que calcula el numero de pasos por batch a realizar al llamar a fit para entrenar el modelo
# Doc: https://androidkt.com/how-to-set-steps-per-epoch-validation-steps-and-validation-split-in-kerass-fit-method/
# https://stackoverflow.com/questions/51748514/does-imagedatagenerator-add-more-images-to-my-dataset
def num_steps(data_size, batch_size):
    return data_size / batch_size

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución de la función
# de pérdida en el conjunto de train y en el de validación, y otra
# con la evolución del accuracy en el conjunto de train y en el de
# validación. Es necesario pasarle como parámetro el historial
# del entrenamiento del modelo (lo que devuelven las funciones
# fit() y fit_generator()).

def mostrarEvolucion(hist):

  loss = hist.history['loss']
  val_loss = hist.history['val_loss']
  plt.plot(loss)
  plt.plot(val_loss)
  plt.legend(['Training loss', 'Validation loss'])
  plt.show()

  acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']
  plt.plot(acc)
  plt.plot(val_acc)
  plt.legend(['Training accuracy', 'Validation accuracy'])
  plt.show()


# Funcion que hace una comparacion de los histogramas de la evolución en ¡¡VALIDACIÓN!!!

def comparaHistog(hist, title):
    for i in hist:
        val_loss = i.history['val_loss']
        plt.plot(val_loss)
        
    plt.legend(["Val.loss " + title[i] for i in range(len(hist))])
    plt.show()
    
    for i in hist:
        val_acc = i.history['val_accuracy']
        plt.plot(val_acc)
        
    plt.legend(["Val.Accuracy " + title[i] for i in range(len(hist))])
    plt.show()
    
    




#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

# A completar
# 1.- Incluir  import del tipo de modelo y capas a usar
# 2.- definir model e incluir las capas en él

# Cargar los datos de entrenamiento y test
x_train, y_train, x_test, y_test = cargarImagenes()

# Establecer parametros

input_shape = (32, 32, 3)

porcentaje_validacion = 0.1

# Numero de epocas para entrenar el modelo
epochs = 30
# Tamanio del batch, es decir, numero de muestras que se procesan
# por cada actualizacion del modelo
batch_size = 32

print("----------- EJERCICIO 1 -----------")
print()
print("Creación del modelo")
print()

# Creacion del modelo
# Definimos el modelo como secuencial para que todas las capas de la red vayan
# una detras de otra (de manera secuencial).
model = Sequential()

# Añadir las capas sera tan facil como usar la funcion add para que se añadan
# una detras de otra.

# Convolucion con 6 mascaras de 5x5 
# Usamos el valor de padding como 'valid' para que se realice la convolución
# donde se pueda ajustar el kernel. Por tanto, se perderá un poco de tamaño en 
# la salida. Por tanto pasamos de 32x32x3 a 28x28x6.
model.add(Conv2D(6, kernel_size=(5,5), padding='valid', input_shape=input_shape))

# Activacion no lineal con ReLU
# https://keras.io/api/layers/activations/
model.add(Activation('relu'))

# MaxPooling 2D
model.add(MaxPooling2D(pool_size=(2,2)))

# Convolucion con 16 mascaras de 5x5 
model.add(Conv2D(16, kernel_size=(5,5), padding='valid', input_shape=(14,14,6)))

# Activacion no lineal con ReLU
model.add(Activation('relu'))

# MaxPooling 2D
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar y Fully Connected (Activacion lineal)
# Se convierte el bloque en un array 1D de 5*5*16 = 400 elementos
model.add(Flatten())
# Unimos este vector con una capa densamente conectada de 50 elementos
model.add(Dense(units=50))

# Activacion no lineal con ReLU
model.add(Activation('relu'))

#Se aplica otra capa densamente conectada al vector con 25 unidades
model.add(Dense(units=25))

# Activacion softmax para transformar las salidas de las neuronas en la probabilidad
# de pertenecer a cada clase
model.add(Activation('softmax'))


#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# Definicion del optimizador
# Para poder modificar los parametros del optimizador, es necesario declararlo previamente
# y crear un objeto
# Doc: https://keras.io/api/optimizers/sgd/
optimizer = SGD()

# Compilacion del modelo
# Compile define la funcion de perdida, el optimizador y las metricas
# Funcion de perdida: entropia cruzada, al tratarse de un problema de clasificacion multiclase (probabilistico)
# Optimizador: SGD (gradiente descendente estocastico)
# Metrica: accuracy (precision)
# Doc: https://stackoverflow.com/questions/47995324/does-model-compile-initialize-all-the-weights-and-biases-in-keras-tensorflow
# Funcion de perdida: https://keras.io/api/losses/

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los 
# pesos aleatorios con los que empieza la red, para poder reestablecerlos 
# después y comparar resultados entre no usar mejoras y sí usarlas.
weights = model.get_weights()

# Reestablecer los valores de los pesos antes del siguiente entrenamiento
#model.set_weights(init_weights)

# Funcion que permite ver la descripcion del modelo
# print(model.summary())

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

# Doc: https://keras.io/api/models/model_training_apis/
evolucion = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_split=porcentaje_validacion)

# Incluir función que muestre la evolución del entrenamiento y validación
mostrarEvolucion(evolucion)


input("Pulse una tecla para continuar...")
print()


#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

prediccion = model.evaluate(x_test, y_test, batch_size = batch_size) 

print("Predicción sobre el conjunto de test: ")
print()

#Incluir función que muestre la perdida y accuracy del test
print("TEST ---- Pérdida: ", prediccion[0])
print("TEST ---- Accuracy: ", prediccion[1])

input("Pulse una tecla para continuar...")
print()









#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

print("----------- EJERCICIO 2 -----------")
print()
print("Mejora del modelo")
print()


print("¡¡¡¡¡IMPORTANTE!!!!")
print("Solamente se ejecutará el modelo final para evitar que tome demasiado tiempo la ejecución...")
print("¡Dentro del codigo se podrá encontrar todos los modelos con su entrenamiento comentado!")
print()
input("Pulse una tecla para continuar...")
print()

#########################################################################
#####           PRIMERA MEJORA: NORMALIZACION DE LOS DATOS        #######
#########################################################################

#print("PRIMERA MEJORA: NORMALIZACION DE LOS DATOS ")
#print()

# Doc: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
generator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, 
                               validation_split=porcentaje_validacion)

# Se entrena el generador para calcular las estadisticas de los datos de entrenamiento
# Doc: https://theailearner.com/2019/07/06/keras-imagedatagenerator-normalization-at-validation-and-test-time/
generator.fit(x_train)

# Se restauran los pesos del modelo antes de entrenar
model.set_weights(weights)

# Se crean dos flow para entrenamiento y validación. Los resultados seran proporcionados
# al metodo fit
# El objetivo de esta funcion es cargar el conjunto de datos de imagenes en la memoria y
# generar batch de datos aumentados
# Doc: https://studymachinelearning.com/keras-imagedatagenerator-with-flow/
training_generator = generator.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = generator.flow(x_train, y_train, batch_size=batch_size, subset='validation')

# La funcion fit_generator entrena el modelo con datos generados batch a batch y es ejecutado
# en paralelo para una mayor eficiencia.
# steps_per_epoch indica los pasos que se van a realizar en cada época
# el cual valdrá el tam del conjunto de entrenamiento (90% de los datos de train porque el 10% restante
# es validación) entre el tamaño del batch establecido.
# Doc: https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# https://androidkt.com/how-to-set-steps-per-epoch-validation-steps-and-validation-split-in-kerass-fit-method/
train_size = len(x_train) * (1 - porcentaje_validacion)
valid_size = len(x_train) * porcentaje_validacion

"""
# Se entrena el modelo
evolucion_normaliz = model.fit(training_generator, validation_data = validation_generator, 
                                         steps_per_epoch = num_steps(train_size,batch_size),
                                         epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

# Incluir función que muestre la evolución del entrenamiento y validación
mostrarEvolucion(evolucion_normaliz)
"""

#input("Pulse una tecla para continuar...")

#########################################################################
#####              SEGUNDA MEJORA: AUMENTO DE LOS DATOS           #######
#########################################################################


#print("SEGUNDA MEJORA: AUMENTO DE LOS DATOS ")
#print()


############################
#   HORIZONTAL FLIP
############################



#print("Se añade HORIZONTAL FLIP ")
#print()

generator_hz = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, 
                               validation_split=porcentaje_validacion, horizontal_flip=True)

generator_hz.fit(x_train)

# Se restauran los pesos del modelo antes de entrenar
model.set_weights(weights)

# Se crean dos flow para entrenamiento y validación. Los resultados seran proporcionados
# al metodo fit
training_generator_hz = generator_hz.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator_hz = generator_hz.flow(x_train, y_train, batch_size=batch_size, subset='validation')

"""
# Se entrena el modelo
# https://stackoverflow.com/questions/51748514/does-imagedatagenerator-add-more-images-to-my-dataset
evolucion_hz = model.fit(training_generator_hz, validation_data = validation_generator_hz, 
                                        steps_per_epoch = num_steps(train_size,batch_size),
                                        epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

# Incluir función que muestre la evolución del entrenamiento y validación
mostrarEvolucion(evolucion_hz)
"""

#input("Pulse una tecla para continuar...")

############################
#         ZOOM - 0.2
############################

#print("Se añade ZOOM DE 0.2 ")
#print()

# Se añade Zoom
# Doc: https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

generator_zoom_02 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, 
                               validation_split=porcentaje_validacion, zoom_range = 0.2)

generator_zoom_02.fit(x_train)
model.set_weights(weights)

training_generator_zoom_02 = generator_zoom_02.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator_zoom_02 = generator_zoom_02.flow(x_train, y_train, batch_size=batch_size, subset='validation')
"""
evolucion_zoom_02 = model.fit(training_generator_zoom_02, validation_data = validation_generator_zoom_02, 
                                         steps_per_epoch = num_steps(train_size,batch_size),
                                         epochs=epochs, validation_steps=num_steps(valid_size,batch_size))
mostrarEvolucion(evolucion_zoom_02)
"""

#input("Pulse una tecla para continuar...")

############################
#         ZOOM - 0.3
############################

#print("Se añade ZOOM DE 0.3 ")
#print()


generator_zoom_03 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, 
                               validation_split=porcentaje_validacion, zoom_range = 0.3)
generator_zoom_03.fit(x_train)
model.set_weights(weights)

training_generator_zoom_03 = generator_zoom_03.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator_zoom_03 = generator_zoom_03.flow(x_train, y_train, batch_size=batch_size, subset='validation')

"""
evolucion_zoom_03 = model.fit(training_generator_zoom_03, validation_data = validation_generator_zoom_03, 
                                         steps_per_epoch = num_steps(train_size,batch_size),
                                         epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion_zoom_03)
"""

#input("Pulse una tecla para continuar...")


############################
#         ZOOM - 0.5
############################

#print("Se añade ZOOM DE 0.5 ")
#print()

generator_zoom_05 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, 
                               validation_split=porcentaje_validacion, zoom_range = 0.5)

generator_zoom_05.fit(x_train)
model.set_weights(weights)

training_generator_zoom_05 = generator_zoom_05.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator_zoom_05 = generator_zoom_05.flow(x_train, y_train, batch_size=batch_size, subset='validation')
"""
evolucion_zoom_05 = model.fit(training_generator_zoom_05, validation_data = validation_generator_zoom_05, 
                                       steps_per_epoch = num_steps(train_size,batch_size),
                                       epochs=epochs, validation_steps=num_steps(valid_size,batch_size))
mostrarEvolucion(evolucion_zoom_05)
"""

#input("Pulse una tecla para continuar...")

"""
print("Comparación entre los resultados obtenidos para ZOOM en VALIDACIÓN ")
print()

histogramas = [evolucion_normaliz, evolucion_zoom_02, evolucion_zoom_03, evolucion_zoom_05 ]
titles = ["Sin zoom", "Zoom: 0.2", "Zoom 0.3", "Zoom 0.5"]

comparaHistog(histogramas, titles)
"""


#input("Pulse una tecla para continuar...")



#################################
#   HORIZONTAL FLIP + ZOOM - 0.2
#################################

#print("Se añade HORIZONTAL FLIP + ZOOM DE 0.2 ")
#print()

generator_hz_zoom = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, 
                               validation_split=porcentaje_validacion, horizontal_flip=True, zoom_range = 0.2)

generator_hz_zoom.fit(x_train)

# Se restauran los pesos del modelo antes de entrenar
model.set_weights(weights)

# Se crean dos flow para entrenamiento y validación. Los resultados seran proporcionados
# al metodo fit_generator
training_generator_hz_zoom = generator_hz_zoom.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator_hz_zoom = generator_hz_zoom.flow(x_train, y_train, batch_size=batch_size, subset='validation')
"""
# Se entrena el modelo
evolucion_hz_zoom = model.fit(training_generator_hz_zoom, validation_data = validation_generator_hz_zoom, 
                                         steps_per_epoch = num_steps(train_size,batch_size),
                                         epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

# Incluir función que muestre la evolución del entrenamiento y validación
mostrarEvolucion(evolucion_hz_zoom)
"""

#input("Pulse una tecla para continuar...")

# Se realiza una comparativa entre los resultados obtenidos en validación para los distintos
# tipos de aumentos de datos

"""
print("Comparación entre los resultados obtenidos con distintas tecnicas de aumento de datos en VALIDACIÓN ")
print()

histogramas = [evolucion_normaliz, evolucion_hz, evolucion_zoom_02, evolucion_hz_zoom]
titles = ["Sin cambios", "Horizontal Flip", "Zoom", "H.Flip + Zoom"]

comparaHistog(histogramas, titles)
"""

#input("Pulse una tecla para continuar...")

#########################################################################
#####                TERCERA MEJORA: RED MAS PROFUNDA             #######
#########################################################################


#print("TERCERA MEJORA: RED MAS PROFUNDA")
#print()

#print("Primera propuesta de arquitectura")
#print()

model1 = Sequential()

model1.add(Conv2D(16, kernel_size=(3,3), padding='valid', input_shape=input_shape))
model1.add(Activation('relu'))
model1.add(Conv2D(32, kernel_size=(3,3), padding='valid'))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model1.add(Activation('relu'))
model1.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(units=150))
model1.add(Activation('relu'))
model1.add(Dense(units=25))
model1.add(Activation('softmax'))

model1.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

weights1 = model1.get_weights()
"""
evolucion1 = model1.fit(training_generator, validation_data = validation_generator,
                        steps_per_epoch = num_steps(train_size,batch_size),
                        epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion1)
"""


#input("Pulse una tecla para continuar...")

#print("Segunda propuesta de arquitectura")
#print()
 
model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3,3), padding='valid', input_shape=input_shape))
model2.add(Activation('relu'))
model2.add(Conv2D(64, kernel_size=(3,3), padding='valid'))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model2.add(Activation('relu'))
model2.add(Conv2D(256, kernel_size=(3, 3), padding='valid'))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Flatten())
model2.add(Dense(units=500))
model2.add(Activation('relu'))
model2.add(Dense(units=150))
model2.add(Activation('relu'))
model2.add(Dense(units=25))
model2.add(Activation('softmax'))

model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

weights2 = model2.get_weights()
"""
evolucion2 = model2.fit(training_generator, validation_data = validation_generator, 
                        steps_per_epoch = num_steps(train_size,batch_size),
                        epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion2)
"""

#input("Pulse una tecla para continuar...")



#########################################################################
#####         CUARTA MEJORA: REGULARIZACIÓN CON DROPOUT           #######
#########################################################################


# Generalmente se usa un dropout de entre el 20%-50% de neuronas
# ya que una probabilidad muy baja tiene efecto minimo y un valor muy grande
# puede resultar en underfitting

#print("CUARTA MEJORA: REGULARIZACIÓN CON DROPOUT")
#print()

#print("Primera propuesta de arquitectura + Dropout")
#print()

model1_dp = Sequential()

model1_dp.add(Conv2D(16, kernel_size=(3,3), padding='valid', input_shape=input_shape))
model1_dp.add(Activation('relu'))
model1_dp.add(Conv2D(32, kernel_size=(3,3), padding='valid'))
model1_dp.add(Activation('relu'))
model1_dp.add(MaxPooling2D(pool_size=(2, 2)))
model1_dp.add(Dropout(0.25))

model1_dp.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model1_dp.add(Activation('relu'))
model1_dp.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model1_dp.add(Activation('relu'))
model1_dp.add(MaxPooling2D(pool_size=(2, 2)))
model1_dp.add(Dropout(0.5))

model1_dp.add(Flatten())
model1_dp.add(Dense(units=150))
model1_dp.add(Activation('relu'))
model1_dp.add(Dropout(0.25))
model1_dp.add(Dense(units=25))
model1_dp.add(Activation('softmax'))

model1_dp.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

weights1_dp = model1_dp.get_weights()

"""
evolucion1_dp = model1_dp.fit(training_generator, validation_data = validation_generator,
                              steps_per_epoch = num_steps(train_size,batch_size),
                              epochs=epochs, validation_steps=num_steps(valid_size,batch_size))
mostrarEvolucion(evolucion1_dp)
"""

#input("Pulse una tecla para continuar...")

#print("Segunda propuesta de arquitectura + Dropout")
#print()

model2_dp = Sequential()

model2_dp.add(Conv2D(32, kernel_size=(3,3), padding='valid', input_shape=input_shape))
model2_dp.add(Activation('relu'))
model2_dp.add(Conv2D(64, kernel_size=(3,3), padding='valid'))
model2_dp.add(Activation('relu'))
model2_dp.add(MaxPooling2D(pool_size=(2, 2)))
model2_dp.add(Dropout(0.25))

model2_dp.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model2_dp.add(Activation('relu'))
model2_dp.add(Conv2D(256, kernel_size=(3, 3), padding='valid'))
model2_dp.add(Activation('relu'))
model2_dp.add(MaxPooling2D(pool_size=(2, 2)))
model2_dp.add(Dropout(0.5))

model2_dp.add(Flatten())
model2_dp.add(Dense(units=500))
model2_dp.add(Activation('relu'))
model2_dp.add(Dropout(0.5))
model2_dp.add(Dense(units=150))
model2_dp.add(Activation('relu'))
model2_dp.add(Dropout(0.25))
model2_dp.add(Dense(units=25))
model2_dp.add(Activation('softmax'))

model2_dp.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
weights2_dp = model2_dp.get_weights()

"""
evolucion2_dp = model2_dp.fit(training_generator, validation_data = validation_generator, 
                              steps_per_epoch = num_steps(train_size,batch_size),
                              epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion2_dp)
"""

#input("Pulse una tecla para continuar...")



#print("AUMENTAMOS EL NÚMERO DE ÉPOCAS")
#print()

epochs = 60

#print("Primera propuesta de arquitectura + Dropout + " + str(epochs) + " epocas")
#print()

model1_dp = Sequential()

model1_dp.add(Conv2D(16, kernel_size=(3,3), padding='valid', input_shape=input_shape))
model1_dp.add(Activation('relu'))
model1_dp.add(Conv2D(32, kernel_size=(3,3), padding='valid'))
model1_dp.add(Activation('relu'))
model1_dp.add(MaxPooling2D(pool_size=(2, 2)))
model1_dp.add(Dropout(0.25))

model1_dp.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model1_dp.add(Activation('relu'))
model1_dp.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model1_dp.add(Activation('relu'))
model1_dp.add(MaxPooling2D(pool_size=(2, 2)))
model1_dp.add(Dropout(0.5))

model1_dp.add(Flatten())
model1_dp.add(Dense(units=150))
model1_dp.add(Activation('relu'))
model1_dp.add(Dropout(0.25))
model1_dp.add(Dense(units=25))
model1_dp.add(Activation('softmax'))

model1_dp.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

weights1_dp = model1_dp.get_weights()

"""
evolucion1_dp = model1_dp.fit(training_generator, validation_data = validation_generator,
                              steps_per_epoch = num_steps(train_size,batch_size),
                              epochs=epochs, validation_steps=num_steps(valid_size,batch_size))
mostrarEvolucion(evolucion1_dp)
"""

#input("Pulse una tecla para continuar...")

#Print("Primera propuesta de arquitectura + Dropout + " + str(epochs) + " epocas")
#print()

model2_dp = Sequential()

model2_dp.add(Conv2D(32, kernel_size=(3,3), padding='valid', input_shape=input_shape))
model2_dp.add(Activation('relu'))
model2_dp.add(Conv2D(64, kernel_size=(3,3), padding='valid'))
model2_dp.add(Activation('relu'))
model2_dp.add(MaxPooling2D(pool_size=(2, 2)))
model2_dp.add(Dropout(0.25))

model2_dp.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model2_dp.add(Activation('relu'))
model2_dp.add(Conv2D(256, kernel_size=(3, 3), padding='valid'))
model2_dp.add(Activation('relu'))
model2_dp.add(MaxPooling2D(pool_size=(2, 2)))
model2_dp.add(Dropout(0.5))

model2_dp.add(Flatten())
model2_dp.add(Dense(units=500))
model2_dp.add(Activation('relu'))
model2_dp.add(Dropout(0.5))
model2_dp.add(Dense(units=150))
model2_dp.add(Activation('relu'))
model2_dp.add(Dropout(0.25))
model2_dp.add(Dense(units=25))
model2_dp.add(Activation('softmax'))

model2_dp.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
weights2_dp = model2_dp.get_weights()

"""
evolucion2_dp = model2_dp.fit(training_generator, validation_data = validation_generator, 
                              steps_per_epoch = num_steps(train_size,batch_size),
                              epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion2_dp)
"""






#########################################################################
#####             QUINTA MEJORA: BATCH NORMALIZATION              #######
#########################################################################


#print("QUINTA MEJORA: BATCH NORMALIZATION")
#print()

#print("Batch Normalization ANTES de la capa de activación")
#print()

print("Ejecución del modelo final")
print()

model_bn = Sequential()

model_bn.add(Conv2D(32, kernel_size=(3,3), padding='valid', input_shape=input_shape))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Conv2D(64, kernel_size=(3,3), padding='valid'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(MaxPooling2D(pool_size=(2, 2)))
model_bn.add(Dropout(0.25))

model_bn.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Conv2D(256, kernel_size=(3, 3), padding='valid'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(MaxPooling2D(pool_size=(2, 2)))
model_bn.add(Dropout(0.5))

model_bn.add(Flatten())
model_bn.add(Dense(units=500))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Dropout(0.5))
model_bn.add(Dense(units=150))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Dropout(0.25))
model_bn.add(Dense(units=25))
model_bn.add(Activation('softmax'))

model_bn.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

weights_bn = model_bn.get_weights()

#print(model_bn.summary())


evolucion_bn = model_bn.fit(training_generator_hz, validation_data = validation_generator_hz, 
                            steps_per_epoch = num_steps(train_size,batch_size),
                            epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion_bn)

input("Pulse una tecla para continuar...")


#print("Batch Normalization DESPUES de la capa de activación")
#print()

model_bn2 = Sequential()


model_bn2.add(Conv2D(32, kernel_size=(3,3), padding='valid', input_shape=input_shape))
model_bn2.add(Activation('relu'))
model_bn2.add(BatchNormalization())
model_bn2.add(Conv2D(64, kernel_size=(3,3), padding='valid'))
model_bn2.add(Activation('relu'))
model_bn2.add(BatchNormalization())
model_bn2.add(MaxPooling2D(pool_size=(2, 2)))
model_bn2.add(Dropout(0.25))

model_bn2.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model_bn2.add(Activation('relu'))
model_bn2.add(BatchNormalization())
model_bn2.add(Conv2D(256, kernel_size=(3, 3), padding='valid'))
model_bn2.add(Activation('relu'))
model_bn2.add(BatchNormalization())
model_bn2.add(MaxPooling2D(pool_size=(2, 2)))
model_bn2.add(Dropout(0.5))


model_bn2.add(Flatten())
model_bn2.add(Dense(units=500))
model_bn2.add(Activation('relu'))
model_bn2.add(BatchNormalization())
model_bn2.add(Dropout(0.5))
model_bn2.add(Dense(units=150))
model_bn2.add(Activation('relu'))
model_bn2.add(BatchNormalization())
model_bn2.add(Dropout(0.25))
model_bn2.add(Dense(units=25))
model_bn2.add(Activation('softmax'))

model_bn2.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

weights_bn2 = model_bn2.get_weights()

"""
evolucion_bn2 = model_bn2.fit(training_generator_hz, validation_data = validation_generator_hz, 
                              steps_per_epoch = num_steps(train_size,batch_size),
                              epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion_bn2)
"""

#input("Pulse una tecla para continuar...")

#########################################################################
#####            PREDICCIÓN PARA EL CONJUNTO DE TEST              #######
#########################################################################

# Se crea un generador para test
# NO AÑADIR SLIPT PARA VALIDACIÓN
# NO AUMENTO DE DATOS PARA TEST

generator_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
generator_test.fit(x_train)


prediccion = model_bn.evaluate(generator_test.flow(x_test, y_test, batch_size = 1, shuffle = False), steps = len(x_test)) 

print("Predicción sobre el conjunto de test: ")
print()



#Incluir función que muestre la perdida y accuracy del test
print("TEST ---- Pérdida: ", prediccion[0])
print("TEST ---- Accuracy: ", prediccion[1])


input("Pulse una tecla para continuar...")


print("----------- BONUS -----------")
print()

print("¡¡¡¡¡IMPORTANTE!!!!")
print("Solamente se ejecutará el modelo final para evitar que tome demasiado tiempo la ejecución...")
print("¡Dentro del codigo se podrá encontrar todos los modelos con su entrenamiento comentado!")
print()
input("Pulse una tecla para continuar...")
print()


#print("OPTIMIZADOR ADAGRAD")
#print()

model_bn.set_weights(weights_bn)

optimizer = Adagrad(learning_rate=0.001,
    initial_accumulator_value=0.1,
    epsilon=1e-07,
    name="Adagrad")

model_bn.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#weights_bn = model_bn.get_weights()
"""
evolucion_adagrad = model_bn.fit(training_generator_hz, validation_data = validation_generator_hz, 
                            steps_per_epoch = num_steps(train_size,batch_size),
                            epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion_adagrad)
"""

#input("Pulse una tecla para continuar...")

#print("OPTIMIZADOR RMSPROP")
#print()
model_bn.set_weights(weights_bn)

optimizer = RMSprop(learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop")

model_bn.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#weights_bn = model_bn.get_weights()
"""
evolucion_rmsprop = model_bn.fit(training_generator_hz, validation_data = validation_generator_hz, 
                            steps_per_epoch = num_steps(train_size,batch_size),
                            epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion_rmsprop)
"""

#input("Pulse una tecla para continuar...")

#print("OPTIMIZADOR ADADELTA")
#print()

model_bn.set_weights(weights_bn)

optimizer = Adadelta(learning_rate=0.001, 
                     rho=0.95, 
                     epsilon=1e-07, 
                     name="Adadelta")

model_bn.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#weights_bn = model_bn.get_weights()

"""
evolucion_adadelta = model_bn.fit(training_generator_hz, validation_data = validation_generator_hz, 
                            steps_per_epoch = num_steps(train_size,batch_size),
                            epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion_adadelta)
"""

#input("Pulse una tecla para continuar...")

#print("OPTIMIZADOR ADAM")
#print()

model_bn.set_weights(weights_bn)

optimizer = Adam(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam")

model_bn.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#weights_bn = model_bn.get_weights()
"""
evolucion_adam = model_bn.fit(training_generator_hz, validation_data = validation_generator_hz, 
                            steps_per_epoch = num_steps(train_size,batch_size),
                            epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion_adam)

input("Pulse una tecla para continuar...")

print("Comparación entre los resultados obtenidos para ZOOM en VALIDACIÓN ")
print()

histogramas = [evolucion_bn, evolucion_adagrad, evolucion_rmsprop, evolucion_adadelta, evolucion_adam ]
titles = ["SGD", "AdaGrad", "RMSProp", "AdaDelta", "Adam"]

comparaHistog(histogramas, titles)

input("Pulse una tecla para continuar...")
"""

input("Ejecución del modelo final --- BONUS")
print()

# Establecemos el mismo modelo que el elegido como mejora del ejercicoi 2
# Se hara usando el optimizador Adam
# A DIFERENCIA del pasado ejercicio, aumentaremos la probabilidad de Dropout
# para paliar con el overfitting producido por el cambio de optimizador.

model_bn = Sequential()

model_bn.add(Conv2D(32, kernel_size=(3,3), padding='valid', input_shape=input_shape))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Conv2D(64, kernel_size=(3,3), padding='valid'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(MaxPooling2D(pool_size=(2, 2)))
model_bn.add(Dropout(0.6))

model_bn.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Conv2D(256, kernel_size=(3, 3), padding='valid'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(MaxPooling2D(pool_size=(2, 2)))
model_bn.add(Dropout(0.5))

model_bn.add(Flatten())
model_bn.add(Dense(units=500))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Dropout(0.6))
model_bn.add(Dense(units=150))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Dropout(0.5))
model_bn.add(Dense(units=25))
model_bn.add(Activation('softmax'))

model_bn.set_weights(weights_bn)

optimizer = Adam( learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam")

model_bn.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#weights_bn = model_bn.get_weights()

evolucion_bn = model_bn.fit(training_generator_hz, validation_data = validation_generator_hz, 
                            steps_per_epoch = num_steps(train_size,batch_size),
                            epochs=epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(evolucion_bn)


input("Pulse una tecla para continuar...")

#########################################################################
#####            PREDICCIÓN PARA EL CONJUNTO DE TEST              #######
#########################################################################

# Se crea un generador para test
# NO AÑADIR SLIPT PARA VALIDACIÓN
# NO AUMENTO DE DATOS PARA TEST

generator_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
generator_test.fit(x_train)


prediccion = model_bn.evaluate(generator_test.flow(x_test, y_test, batch_size = 1, shuffle = False), steps = len(x_test)) 

print("Predicción sobre el conjunto de test: ")
print()



#Incluir función que muestre la perdida y accuracy del test
print("TEST ---- Pérdida: ", prediccion[0])
print("TEST ---- Accuracy: ", prediccion[1])
