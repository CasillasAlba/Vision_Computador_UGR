# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 00:22:16 2021

@author: Alba
"""
#########################################################################
################### OBTENER LA BASE DE DATOS ############################
#########################################################################

# Descargar las imágenes de http://www.vision.caltech.edu/visipedia/CUB-200.html
# Descomprimir el fichero.
# Descargar también el fichero list.tar.gz, descomprimirlo y guardar los ficheros
# test.txt y train.txt dentro de la carpeta de imágenes anterior. Estos 
# dos ficheros contienen la partición en train y test del conjunto de datos.

#########################################################################
################ CARGAR LAS LIBRERÍAS NECESARIAS ########################
#########################################################################

# Terminar de rellenar este bloque con lo que vaya haciendo falta

# Importar librerías necesarias
import numpy as np
import keras
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, GlobalAveragePooling2D, Activation, BatchNormalization

# Importar el optimizador a usar
from tensorflow.keras.optimizers import SGD

# Importar modelos y capas específicas que se van a usar


# Importar el modelo ResNet50 y su respectiva función de preprocesamiento,
# que es necesario pasarle a las imágenes para usar este modelo
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator


#########################################################################
################## FUNCIÓN PARA LEER LAS IMÁGENES #######################
#########################################################################

# Dado un fichero train.txt o test.txt y el path donde se encuentran los
# ficheros y las imágenes, esta función lee las imágenes
# especificadas en ese fichero y devuelve las imágenes en un vector y 
# sus clases en otro.

def leerImagenes(vec_imagenes, path):
  clases = np.array([img.split('/')[0] for img in vec_imagenes])
  imagenes = np.array([img_to_array(load_img(path + "/" + img, 
                                             target_size = (224, 224))) 
                       for img in vec_imagenes])
  return imagenes, clases

#########################################################################
############# FUNCIÓN PARA CARGAR EL CONJUNTO DE DATOS ##################
#########################################################################

# Usando la función anterior, y dado el path donde se encuentran las
# imágenes y los archivos "train.txt" y "test.txt", devuelve las 
# imágenes y las clases de train y test para usarlas con keras
# directamente.

def cargarDatos(path):
  # Cargamos los ficheros
  train_images = np.loadtxt(path + "/train.txt", dtype = str)
  test_images = np.loadtxt(path + "/test.txt", dtype = str)
  
  # Leemos las imágenes con la función anterior
  train, train_clases = leerImagenes(train_images, path)
  test, test_clases = leerImagenes(test_images, path)
  
  # Pasamos los vectores de las clases a matrices 
  # Para ello, primero pasamos las clases a números enteros
  clases_posibles = np.unique(np.copy(train_clases))
  for i in range(len(clases_posibles)):
    train_clases[train_clases == clases_posibles[i]] = i
    test_clases[test_clases == clases_posibles[i]] = i

  # Después, usamos la función to_categorical()
  train_clases = to_categorical(train_clases, 200)
  test_clases = to_categorical(test_clases, 200)
  
  # Barajar los datos
  train_perm = np.random.permutation(len(train))
  train = train[train_perm]
  train_clases = train_clases[train_perm]

  test_perm = np.random.permutation(len(test))
  test = test[test_perm]
  test_clases = test_clases[test_perm]
  
  return train, train_clases, test, test_clases

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


def mostrar_fine_tune(loss, val_loss, acc, val_acc):
  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Accuracy Entrenamiento')
  plt.plot(val_acc, label='Accuracy Validación')
  plt.ylim([0.0, 1])
  plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Comienzo del Fine Tuning')
  plt.legend(loc='lower right')
  plt.title('Evolución del accuracy')
  
  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Pérdida en entrenamiento')
  plt.plot(val_loss, label='Pérdida en validación')
  plt.ylim([0, 5.0])
  plt.plot([initial_epochs-1,initial_epochs-1],
           plt.ylim(), label='Comienzo del Fine Tuning')
  plt.legend(loc='upper right')
  plt.title('Evolución de la función de pérdida')
  plt.xlabel('epoch')
  plt.show()

"""## Usar ResNet50 preentrenada en ImageNet como un extractor de características"""

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos.

ruta = "images"
optimizer = SGD()
epochs = 20
batch_size = 32
porcentaje_validacion = 0.1

x_train, y_train, x_test, y_test = cargarDatos(ruta)

# Se utiliza ImageDataGenerator para pasarle la funcion de preprocesamiento
# importada de la libreria
generador = ImageDataGenerator(preprocessing_function = preprocess_input)

# ¡¡¡ No hace falta usar fit!!! Ya que nuestro ImageDataGenerator aqui no hace
# ningun tipo de normalizacion de datos
# Doc: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
#generator.fit(x_train)

# Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
resnet50 = ResNet50(include_top=False, weights='imagenet',pooling='avg')

#print(resnet50.summary())

# Extraer las características las imágenes con el modelo anterior.
# Para ello, se utilizara el metodo predict, el cual devuelve un array de predicciones sobre
# las imagenes de entrada, cargadas mediante batchs (gracias al método flow y con shuffle=False
# para que no se hagan predicciones incorrectas). Como a nuestro modelo resnet50 le hemos quitado
# la parte clasificadora, el vector que devuelve sera el vector de caracteristicas (2048 caracteristicas)
# Doc: https://keras.io/api/models/model_training_apis/

caract_train = resnet50.predict(generador.flow(x_train, batch_size=1, shuffle=False), steps=len(x_train))

caract_test = resnet50.predict(generador.flow(x_test, batch_size=1, shuffle=False), steps=len(x_test))

print("----------- EJERCICIO 3 -----------")
print()
print("RESNET50 COMO EXTRACTOR DE CARACTERÍSTICAS")
print()


print("-----PRIMER MODELO------")
print()

############################################
#####           APARTADO A           #######
############################################

# Las características extraídas en el paso anterior van a ser la entrada
# de nuestro pequeño modelo de FC, donde la última capa será la que nos
# clasifique las clases de Caltech-UCSD.

# De esta forma, es como si hubieramos fijado todos los parametros de ResNet50
# y estuvieramos entrenando unicamente las capas añadidas.


# Nuestro primer modelo tendrá una única capa Fully Connected adecuada a
# la dimensionalidad de nuestro problema, es decir, 200 clases.

# Se crea el modelo
model_200 = Sequential()
# ¡¡¡¡IMPORTANTE!!! - Se que indicar SI O SI el valor de input_shape para corregir el
# siguiente error:  Weights for model sequential_3 have not yet been created.
# Weights are created when the Model is first called on inputs or `build()` is called with an `input_shape`.
model_200.add(Dense(200, input_shape=(2048,)))
# Activacion softmax para transformar las salidas de las neuronas en la probabilidad
# de pertenecer a cada clase
model_200.add(Activation('softmax'))

# Se compila el modelo
model_200.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#print(model_200.summary())

# Obtener pesos del modelo creado
weights_200 = model_200.get_weights()

# Entrenar modelo
# En la función fit() se puede usar el argumento validation_split
# Se entrena SOLAMENTE nuestro modelo PERO las carcateristicas extraidas de ResNet50!
evolucion = model_200.fit(caract_train, y_train, epochs=epochs, batch_size=batch_size, 
                          validation_split=porcentaje_validacion)

# Mostrar evolucion
mostrarEvolucion(evolucion)

input("Pulse una tecla para continuar...")

# Predecir los datos
prediccion = model_200.evaluate(caract_test, y_test) 


print("Predicción sobre el conjunto de test: ")
print()

#Incluir función que muestre la perdida y accuracy del test
print("TEST ---- Pérdida: ", prediccion[0])
print("TEST ---- Accuracy: ", prediccion[1])


input("Pulse una tecla para continuar...")
print()

##############################################################################
##############################################################################



print("-----SEGUNDO MODELO------")
print()

# Al segundo modelo se le añsdiran mas capas densas, a parte de la FC
# adecuada a la dimensionalidad del problema

# Se crea el modelo
model_2dense = Sequential()

model_2dense.add(Dense(1024, input_shape=(2048,)))
model_2dense.add(Activation('relu'))
model_2dense.add(Dropout(0.4))
model_2dense.add(Dense(200))
model_2dense.add(Activation('softmax'))

# Se compila el modelo
model_2dense.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#print(model_2dense.summary())

# Obtener pesos del modelo creado
weights_2dense = model_2dense.get_weights()

# Entrenar modelo
# En la función fit() se puede usar el argumento validation_split
evolucion2 = model_2dense.fit(caract_train, y_train, epochs=epochs, batch_size=batch_size, 
                          validation_split=porcentaje_validacion)

# Mostrar evolucion
mostrarEvolucion(evolucion2)

input("Pulse una tecla para continuar...")

# Predecir los datos
prediccion = model_2dense.evaluate(caract_test, y_test) 

print("Predicción sobre el conjunto de test: ")
print()

#Incluir función que muestre la perdida y accuracy del test
print("TEST ---- Pérdida: ", prediccion[0])
print("TEST ---- Accuracy: ", prediccion[1])

input("Pulse una tecla para continuar...")
print()




############################################
#####           APARTADO B           #######
############################################

print("-----TERCER MODELO MODELO------")
print()

# Definir el modelo ResNet50 (preentrenado en ImageNet y sin las capas FC ni AveragePooling).
resnet50_nopool = ResNet50(include_top=False, weights='imagenet', pooling=None)

# Extraer las características las imágenes con el modelo anterior.
caract_train_nopool = resnet50_nopool.predict(generador.flow(x_train, batch_size=1, shuffle=False),
                                steps=len(x_train))

caract_test_nopool = resnet50_nopool.predict(generador.flow(x_test, batch_size=1, shuffle=False),
                               steps=len(x_test))

model_nopool = Sequential()


model_nopool.add(Conv2D(512, kernel_size = (3, 3), input_shape = (7, 7, 2048)))
model_nopool.add(Activation('relu'))
model_nopool.add(BatchNormalization())
model_nopool.add(GlobalAveragePooling2D())
model_nopool.add(Dropout(0.4))

model_nopool.add(Flatten())
model_nopool.add(Dense(1024))
model_nopool.add(Activation('relu'))
model_nopool.add(BatchNormalization())
model_nopool.add(Dropout(0.4))
model_nopool.add(Dense(200))
model_nopool.add(Activation('softmax'))


# Se compila el modelo
model_nopool.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#print(model_nopool.summary())

# Obtener pesos del modelo creado
weights_nopool = model_nopool.get_weights()

# Entrenar modelo
# En la función fit() se puede usar el argumento validation_split
evolucion3 = model_nopool.fit(caract_train_nopool, y_train, epochs=epochs, batch_size=batch_size, 
                          validation_split=porcentaje_validacion)

# Mostrar evolucion
mostrarEvolucion(evolucion3)

input("Pulse una tecla para continuar...")

# Predecir los datos
prediccion = model_nopool.evaluate(caract_test_nopool, y_test) 

print("Predicción sobre el conjunto de test: ")
print()

#Incluir función que muestre la perdida y accuracy del test
print("TEST ---- Pérdida: ", prediccion[0])
print("TEST ---- Accuracy: ", prediccion[1])

input("Pulse una tecla para continuar...")
print()


"""## Reentrenar ResNet50 (fine tunning)"""

print("RESNET50 COMO AJUSTE REFINADO")
print()

# ESTE EJERCICIO HA SIDO RESUELTO SIGUIENDO LOS PASOS DE LA DOCUMENTACIÓN DE TENSORFLOW:
# Doc: https://www.tensorflow.org/tutorials/images/transfer_learning

print("Se extraen primero características...")
print()

# Se congela el modelo, evitando que los pesos deuna capa
# determinada se actualicen durante el entrenamiento
# Nosotros estamos congelando la base convolucional
resnet50.traineable = False


initial_epochs = 10

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos.
generador_train = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=porcentaje_validacion)

generador_test = ImageDataGenerator(preprocessing_function=preprocess_input)

# Añadir nuevas capas al final de ResNet50 (recuerda que es una instancia de
# la clase Model).

salida_resnet = resnet50.output
salida_resnet = Dense(1024) (salida_resnet)
salida_resnet = Activation("relu") (salida_resnet)
salida_resnet = Dropout(0.4) (salida_resnet)
salida_resnet = Dense(200) (salida_resnet)
salida_resnet = Activation("softmax") (salida_resnet)

model = Model(inputs = resnet50.input, outputs = salida_resnet)


# Compilación y entrenamiento del modelo.
# A completar.

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])


# Se crean dos flow para entrenamiento y validación. Los resultados seran proporcionados
# al metodo fit
training_generator = generador_train.flow(x_train, y_train, batch_size = batch_size, subset = "training")
validation_generator = generador_train.flow(x_train, y_train, batch_size = batch_size, subset = "validation")

# La funcion fit_generator entrena el modelo con datos generados batch a batch y es ejecutado
# en paralelo para una mayor eficiencia.
# steps_per_epoch indica los pasos que se van a realizar en cada época
# el cual valdrá el tam del conjunto de entrenamiento (90% de los datos de train porque el 10% restante
# es validación) entre el tamaño del batch establecido.
# Doc: https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# https://androidkt.com/how-to-set-steps-per-epoch-validation-steps-and-validation-split-in-kerass-fit-method/
train_size = len(x_train) * (1 - porcentaje_validacion)
valid_size = len(x_train) * porcentaje_validacion

# Se entrena la CAPA SUPERIOR del modelo
# Haciendo esto ahora mismo ResNet50 estaria actuando solamente como un extractor de caracteristicas
fit_top_layer = model.fit(training_generator, validation_data = validation_generator, 
                                         steps_per_epoch = num_steps(train_size,batch_size),
                                         epochs=initial_epochs, validation_steps=num_steps(valid_size,batch_size))

mostrarEvolucion(fit_top_layer)

input("Pulse una tecla para continuar...")

prediccion = model.evaluate(generador_test.flow(x_test, y_test, batch_size = 1, shuffle = False), steps = len(x_test)) 

print("Predicción sobre el conjunto de test: ")
print()

#Incluir función que muestre la perdida y accuracy del test
print("TEST ---- Pérdida: ", prediccion[0])
print("TEST ---- Accuracy: ", prediccion[1])

input("Pulse una tecla para continuar...")

print("Se refina el modelo...")
print()

# Finalmente, descongelamos la base del modelo y entrenamos todo el modelo end-to-end con una tasa de aprendizaje baja

# Cabe destacar que, aunque el modelo base se vuelve entrenable, todavia se esta ejecutando en modeo
# de inferencia, ya que pasamos training = false cuando lo llamamos al construir el modelo.

# Se descongela el modelo
resnet50.traineable = True

# Se vuelve a compilar para que los cambios surjan efecto
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

# Se reanuda el entrenamiento
fine_tune= model.fit(training_generator, validation_data = validation_generator, 
                                         steps_per_epoch = num_steps(train_size,batch_size),
                                         initial_epoch= fit_top_layer.epoch[-1],
                                         epochs=total_epochs, validation_steps=num_steps(valid_size,batch_size))

# Se calcua el valor de perdida y accuracy para entrenamiento y validacion

loss = fit_top_layer.history['loss'] + fine_tune.history['loss']
val_loss = fit_top_layer.history['val_loss'] + fine_tune.history['val_loss']

acc = fit_top_layer.history['accuracy'] + fine_tune.history['accuracy']
val_acc = fit_top_layer.history['val_accuracy'] + fine_tune.history['val_accuracy']
   
# Se muestran la evolucion donde se podra observar los efectos del fine-tuning
mostrar_fine_tune(loss, val_loss, acc, val_acc)

input("Pulse una tecla para continuar...")

prediccion = model.evaluate(generador_test.flow(x_test, y_test, batch_size = 1, shuffle = False), steps = len(x_test)) 

print("Predicción sobre el conjunto de test: ")
print()

#Incluir función que muestre la perdida y accuracy del test
print("TEST ---- Pérdida: ", prediccion[0])
print("TEST ---- Accuracy: ", prediccion[1])