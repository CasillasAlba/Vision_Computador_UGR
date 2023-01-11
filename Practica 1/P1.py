# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:50:44 2021

@author: Alba Casillas Rodríguez
"""

"""

Se importan las libererías necesarias para la práctica:
    
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import warnings
warnings.filterwarnings("ignore")

""" 

    FUNCIONES AUXILIARES

"""

# Función que lee una imagen a partir de un fichero. En esta práctica leeremos las imagenes en escala de grises (flagColor)

def leeimagen(filename, flagColor):
    # Cargamos la imagen
    im = cv2.imread(filename,flagColor) 
    
    return im



# Pasamos las imágenes a float64 después de cargarlas para tener una mayor precisión a la hora de realizar
# operaciones con ellas. Para ello usamos el parámetro astype(np.float64).

def img_a_float(img):
    img_transf = np.copy(img)
    img_transf = img_transf.astype(np.float64)
    
    return img_transf



# Función que normaliza una imagen
 
def normaliza_imagen(im):
    
    # Para no trabajar sobre la imagen original, haremos una copia
    im_normalized = np.copy(im)
    
    min = im_normalized.min()
    max = im_normalized.max()
    
    # Formula para normalizar una matriz:
    # Fuente: https://towardsdatascience.com/everything-you-need-to-know-about-min-max-normalization-in-python-b79592732b79
    
    im_normalized = (im_normalized - min) / (max - min)
          
    return im_normalized     



# Función que muestra una imagen con los métodos de openCV

def mostrarimagen(img, titulo="Imagen"):
    
    #cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.imshow(titulo,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Función que grafica las máscaras del ejercicio 1A mediante el uso de matplotlib

def mostrar_mascaras(i, gauss, deriv1, deriv2, legend):
    
    titulo = "Mascaras de tamaño " + str(i)
    plt.title(titulo)
    
    plt.plot(gauss)
    plt.plot(deriv1)
    plt.plot(deriv2)
    plt.legend([legend[0], legend[1] , legend[2]], loc= "lower right", fontsize = "x-small")
    
    plt.show()

        
  
# Función que concatena una lista de imagenes mediante el uso de hconcat visto en la práctica 0.
# Normalizamos primero las imagenes a concatenar 
# Utilizaremos este método para visualizar las pirámides gaussiana y laplaciana
        
def pintaMI(vim):
    
    # Se obtiene la dimensión máxima
    # Función shape: El "shape" de un array es una tupla con el numero de elementos por eje (dimensión)
    # https://www.python-course.eu/numpy_create_arrays.php
    max_height = vim[0].shape[0]
    vim_copy = np.copy(vim)
    
    piramide = []
    piramide.append(normaliza_imagen(vim_copy[0]))

    
    for i in range(1, len(vim)):   

        # https://stackoverflow.com/questions/36255654/how-to-add-border-around-an-image-in-opencv-python
        # https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36
        vim_copy[i] = normaliza_imagen(vim_copy[i])
        vim_copy[i] = cv2.copyMakeBorder(vim_copy[i], 0, max_height - vim_copy[i].shape[0] ,0,0,cv2.BORDER_CONSTANT, value = (0,0,0,0))
      
        piramide.append(vim_copy[i])
    
    vim_concat = cv2.hconcat(piramide)
    mostrarimagen(vim_concat, "Concatenacion")


# Representa varias imagenes con sus títulos en una misma ventana.

def pintaConTitulo(vim, titles):
    num_im = len(vim)
    
    # Creamos un subplot por cada imagen
    # "_" sirve para ignorar el primer caracter (en este caso, la figura en sí)
    # Si no lo añadiesemos se obtendria el error: 'Figure' object has no attribute 'imshow'
    _, list_subplots = plt.subplots(1, num_im)  
    
    for i in range(num_im):
       list_subplots[i].imshow(vim[i], cmap="gray") 
       list_subplots[i].set_title(titles[i])
    
       # list_subplots[i].xticks([]), list_subplots[i].yticks([])
       list_subplots[i].axis("off")
        
    plt.show()
    
    

"""

    FUNCIONES AUXILIARES EJERCICIO 1A

"""

"""

Calculamos los valores de sigma y tam_masc despejando los valores de la fórmula:
    T = 2*[3*sigma] + 1

"""

def calculo_sigma(tam_masc):
    # Usamos //2 para que el valor de k sea un valor entero
    k = (tam_masc - 1)//2
    sigma = k/3 
    
    return sigma, k

def calculo_tam_masc(sigma):
    k = int((3*sigma))
    tam_masc = 2*k + 1
    
    return tam_masc, k


"""

Calculamos la función gaussiana y las derivadas primera y segunda de la gaussiana

"""

def funcion_gaussiana(x, sigma):
    
    return math.exp(- (x**2) / (2 * (sigma**2)))

def funcion_primera_deriv_gaussiana(x, sigma):
    
    gaussian = funcion_gaussiana(x, sigma)

    return ( - (gaussian * x ) / (sigma**2) )

def funcion_segunda_deriv_gaussiana(x, sigma):
    
    gaussian = funcion_gaussiana(x, sigma)
    
    sigma2 = sigma**2
    
    return ( - ( ((sigma**2) - (x**2) ) * gaussian) / (sigma2 * sigma2) )


"""

Calculamos la máscara indicada por el parámetro "funcion", mediante el sigma
o tam_masc pasados como argumento

"""

def masc_gaussiana(funcion, tam_masc=None, sigma=None):
    
    masc = []
    
    # Calculamos sigma o tam_masc dependiendo del valor dado y la correspondiente k
    if sigma == None:
        sigma, k = calculo_sigma(tam_masc)
    else:
        tam_masc, k = calculo_tam_masc(sigma)

    # Calculamos el rango de la máscara que irá de [-k,k]
    # Usamos"k+1" porque la función np.arange proporciona un intervalo abierto por la derecha: [-k, k)
    # Doc: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    rango_masc = np.arange(-k, k+1)
   
    for i in rango_masc:
        masc.append(funcion(i,sigma))
        
    # Si la funcion es la gaussiana se debe normalizar -> la máscara debe sumar 1 (dividir masc / sum(masc))
    # Las derivadas no se normalizan porque la suma de los valores de la mascara debe valer 0.
    
    if funcion == funcion_gaussiana:
        masc_norm = masc / np.sum(masc)
    else:
        masc_norm = masc
    
    return masc_norm




"""

    EJERCICIO 1A
    
"""

def ejercicio1A():
    
    # Creamos una lista con los tamanios de mascara deseados para luego iterar sobre ella
    tam_masc = [5,7,9]
    
    for i in tam_masc:
    
        
        print("TAMAÑO DE MÁSCARA:", i)
        print()
        
        masc_gauss =  masc_gaussiana(funcion_gaussiana, i, None)
        print("Máscara obtenida de la función gaussiana: ")
        print(masc_gauss)
        
        print()
        
        masc_deriv1 = masc_gaussiana(funcion_primera_deriv_gaussiana, i, None)
        print("Máscara obtenida de la función PRIMERA DERIVADA gaussiana: ")
        print(masc_deriv1)
        
        print()
        
        masc_deriv2 = masc_gaussiana(funcion_segunda_deriv_gaussiana, i, None)
        print("Máscara obtenida de la función SEGUNDA DERIVADA gaussiana: ")
        print(masc_deriv2)
        
        # Mostramos las mascaras discretas 1D       
        legend = ["Gaussiana", "Gaussiana 1ª Deriv", "Gaussiana 2ª Deriv"]
        mostrar_mascaras(i, masc_gauss, masc_deriv1, masc_deriv2, legend)
        
        input("Pulse una tecla para continuar...")
        print()
          
        print("Calculamos las máscaras mediante getDerivKernel de OpenCV ")
        print()
        
        masc_gauss_opencv = cv2.getDerivKernels(0,1,i)
        print("Máscara obtenida de la función gaussiana por OpenCV: ")
        print(masc_gauss_opencv[0])
        masc_gauss_opencv = cv2.getDerivKernels(0,1,i, normalize=True)
        
        
        print()
        
        masc_deriv1_opencv = cv2.getDerivKernels(1,0,i)
        print("Máscara obtenida de la función PRIMERA DERIVADA gaussiana por OpenCV: ")
        print(masc_deriv1_opencv[0])
        masc_deriv1_opencv = cv2.getDerivKernels(1,0,i, normalize=True)
        
        print()
        
        masc_deriv2_opencv = cv2.getDerivKernels(2,0,i)
        print("Máscara obtenida de la función SEGUNDA DERIVADA gaussiana por OpenCV: ")
        print(masc_deriv2_opencv[0])
        masc_deriv2_opencv = cv2.getDerivKernels(2,0,i, normalize=True)
        
        # Mostramos las mascaras discretas 1D  
        legend = ["Gaussiana OpenCV", "Gaussiana 1ª Deriv OpenCV", "Gaussiana 2ª Deriv OpenCV"]
        mostrar_mascaras(i, masc_gauss_opencv[0], masc_deriv1_opencv[0], masc_deriv2_opencv[0], legend)
        
        input("Pulse una tecla para continuar...")
        print()
 
"""

    EJERCICIO 1B
    
"""
    
def ejercicio1B():
    kbinomial_3_deriv = [-1,0,1]
    kbinomial_3 = [1,2,1]
    
    print("Mascaras de alisamiento: ")
    
    print("Convolucionamos un Kernel binomial de alisamiento de tam 3 y consigo mismo")
    kbinomial_5 = np.convolve(kbinomial_3, kbinomial_3)
    print(kbinomial_5)
    print()
   
    print("Convolucionamos un Kernel binomial de alisamiento de tam 5 y el kernel binomial de tam 3 ")
    kbinomial_7 = np.convolve(kbinomial_3, kbinomial_5)
    print(kbinomial_7)
    print()
    
    print("Convolucionamos un Kernel binomial de alisamiento de tam 7 y el kernel binomial de tam 3 ")
    kbinomial_9 = np.convolve(kbinomial_3, kbinomial_7)
    print(kbinomial_9)
    print()
    
        
    input("Pulse una tecla para continuar...")
    
    print("Mascaras de primera derivada: ")
    
    print("Calculamos las mascaras de un Kernel binomial de derivada de tam 3 y la derivada de tam 3 ")
    kbinomial_5_deriv = np.convolve(kbinomial_3_deriv, kbinomial_3)
    print(kbinomial_5_deriv)
    print()
   
    print("Calculamos las mascaras de un Kernel binomial de derivada de tam 5 y la derivada de tam 3 ")
    kbinomial_7_deriv = np.convolve(kbinomial_3_deriv, kbinomial_5)
    print(kbinomial_7_deriv)
    print()
    
    print("Calculamos las mascaras de un Kernel binomial de derivada de tam 7 y la derivada de tam 3 ")
    kbinomial_9_deriv = np.convolve(kbinomial_3_deriv, kbinomial_7)
    print(kbinomial_9_deriv)
    print()
    
    input("Pulse una tecla para continuar...")

    print("Ejecutamos getDerivKernels(0,1,9)")
    
    res = np.transpose(cv2.getDerivKernels(0,1,9))
    print(res)
    
    
"""

    FUNCIONES AUXILIARES EJERCICIO 1C

"""
     
def aniade_padding(img, padding, borde):
    
    # Aniadimos a la imagen de entrada (img), un padding arriba, abajo, izquierda y derecha
    # cuyo tipo de borde puede ser (BORDER_CONSTANT o BORDER_REFLEC) y de valor 0 (el cual indica)
    # que representa el color del borde si este es BORDER_CONSTANT
    # Doc: https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/
    # https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36
    
    imagen_padding = cv2.copyMakeBorder(img, padding, padding, padding, padding, borde, 0)
    
    return np.array(imagen_padding)

     
def elimina_padding(img, height, width, padding):
    
    # Para eliminar el padding creamos, a partir de la imagen con padding, una imagen en forma de 
    # SUBMATRIZ que empieza en el valor del padding (para quitar las primeras "padding" filas/columnas
    # y acabará (el indice al que llega) en el ancho/alto de la imagen original + padding                                       
    return img[padding:(height+padding),padding:(width+padding)]
  
    
def convolucion2D(imagen, hmask, vmask):
    # Guardamos el alto y ancho de la imagen original
    # el bucle tiene que dar tantas vueltas como filas tenga la imagen original, sin el padding
     # Imagen monobanda
    if len(imagen.shape) == 2:
        
        # Guardamos el alto y ancho de la imagen original
        # el bucle tiene que dar tantas vueltas como filas tenga la imagen original, sin el padding
        height_img, width_img = imagen.shape
        
    else:
        # Imagen tribanda (bonus apartado 2)
        height_img, width_img, depth_img = imagen.shape  

    
    # Calculamos el padding que tendrá la imagen con respecto al tamaño de la máscara y se lo añadimos a la iamgen
    padding = int((len(hmask) - 1)/2)
    imagen_padding = aniade_padding(imagen, padding, cv2.BORDER_REFLECT)
    
    # Creamos una matriz de 0 con tamaño de la imagen + padding. En ella guardaremos el resultado de la primera convolución
    matrix_tmp = np.zeros(imagen_padding.shape)      
    
    # Recorremos una vez las filas (por todo el alto de la imagen original) y realizamos la primera convolución 
    # Es por ello que guardamos el resultado en matrix_tmp[inicio+padding], ya que al recorrer la dimensión
    # de la imagen original, debemos "ignorar" el padding a la hora de guardar el resultado.
    # De esta manera, el resultado quedará centrado.
    #
    # Para la multiplicación de la máscara por la matriz de la imagen, se ha hecho uso de la función matmul
    # la cual devuelve la matriz producto de los dos parámetros de entrada, respetando que (n,k),(k,m)->(n,m)
    # De esta forma, si nuestra máscara fuese [1,2,1], la multiplicamos por imagen[i: i+len(mask = 3)]
    # Doc: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
           
    inicio = 0
    fin = inicio + len(hmask)
    
    for i in range(height_img):
        matrix_tmp[inicio+padding] = np.matmul(hmask, imagen_padding[inicio:fin,:])
        inicio = inicio + 1
        fin = fin + 1
    
    # Realizamos la transpuesta de la matriz para la siguiente convolución y sacamos las dimensiones
    matrix_tmp_traspuesta = np.transpose(matrix_tmp)
     # Imagen monobanda
    if len(imagen.shape) == 2:
        
        # Guardamos el alto y ancho de la imagen original
        # el bucle tiene que dar tantas vueltas como filas tenga la imagen original, sin el padding
        height_matrix, width_matrix = matrix_tmp_traspuesta.shape
        
    else:
        # Imagen tribanda (bonus apartado 2)
        height_matrix, width_matrix, depth_matrix = matrix_tmp_traspuesta.shape
    
    # Creamos otra matriz de ceros del tamaño del resultado anterior ya transpuesto
    # ya que las matrices puede que no sean cuadradas
    convolucion = np.zeros(matrix_tmp_traspuesta.shape)
    
    inicio = 0
    fin = inicio + len(vmask)
    
    # Realizamos la segunda convolución, también por filas, de la transpuesta del resultado
    # Ahora el for utiliza (height_matrix - (padding*2)) ya que al usar las dimensiones de la
    # matriz temporal, tenemos que ignorar su padding.
    # La dinámica de esta convolución es la misma que la anterior.
    for i in range((height_matrix - (padding*2))):
        convolucion[inicio+padding] = np.matmul(vmask, matrix_tmp_traspuesta[inicio:fin,:])
        inicio = inicio + 1
        fin = fin + 1
    
    # Realizamos la transpuesta de nuestro resultado para tener la imagen original, pero convolucionada
    convolucion_traspuesta = np.transpose(convolucion)   
    
    # Eliminamos el padding que tenía la imagen
    imgConvolucionada = elimina_padding(convolucion_traspuesta, height_img, width_img , padding)
  
    return imgConvolucionada   
    
"""

    EJERCICIO 1C
    
"""

def ejercicio1C(imagen, v_sigmas):
           
    for i in v_sigmas:
        
        print("Convolución de la imagen para sigma=", i)
        
        hmask = masc_gaussiana(funcion_gaussiana, None, i)
        
        # vmask podría ser una máscara completamente diferente, pero para comprobar
        # el funcionamiento del método usaré la misma que hmask
        vmask = np.copy(hmask)
           
        imgConvolucionada = convolucion2D(imagen, hmask, vmask)
        
        # Normalizamos la imagen resultante y la mostramos
        normalizada = normaliza_imagen(imgConvolucionada)
        titulo = "Convolucion con sigma" + str(i)
        mostrarimagen(normalizada, titulo)
        
        input("Pulse una tecla para continuar...")
        print()

 
def ejercicio1C_A(imagen, v_sigmas):
    
    for i in v_sigmas:
        
        print("Convolución con GaussianBlur de la imagen para sigma=", i)
        
        _, k = calculo_tam_masc(i)
        
        # GaussianBlur de OpenCV recibirá la imagen que convolucionar,
        # el tamaño del kernel (el cual se debe escribir como una tupla, donde el ancho y largo pueden diferir pero siempre
        # deberán ser POSITIVOS E IMPARTES); en nuestro caso utilizamos el mismo valor.
        # El valor de sigma en la dirección X y para la dirección Y pasamos un -1 para que el propio método
        # estime el valor más adecuado.
        
        imagenOpenCV = cv2.GaussianBlur(imagen, ksize=(k,k), sigmaX=i, sigmaY=-1)
        
        # Mostramos los resultados
        titulo = "GaussianBlur con sigma" + str(i)
        mostrarimagen(imagenOpenCV, titulo)
        
        input("Pulse una tecla para continuar...")
        print()


def ejercicio1C_B(imagen, v_sigmas):
    
    for i in v_sigmas:
        print("Derivadas de la imagen para sigma=", i)
        print()
        
        # Calculamos las máscaras para el sigma establecido
        # Un Kernel corresponderá a la función Gaussiana y el otro a la primera derivada de la Gaussiana     
        hmask = masc_gaussiana(funcion_gaussiana, None, i)
        vmask = masc_gaussiana(funcion_primera_deriv_gaussiana, None, i)
        
        print("Derivada respecto de X")
        
        # Gx = I * Gy * G'x
        imgConvolucionada = convolucion2D(imagen, vmask, hmask)
        
        normalizada = normaliza_imagen(imgConvolucionada)
        titulo = "Derivada respecto de X con sigma" + str(i)
        mostrarimagen(normalizada, titulo)
        
        
        input("Pulse una tecla para continuar...")
        print()
        
        print("Derivada respecto de Y")
        
        # Gy = I * Gx * G'y
        imgConvolucionada = convolucion2D(imagen, hmask, vmask)
        
        #Normalizamos y mostramos los resultados
        normalizada = normaliza_imagen(imgConvolucionada)
        titulo = "Derivada respecto de Y con sigma" + str(i)
        mostrarimagen(normalizada, titulo)
             
        input("Pulse una tecla para continuar...")
        print()
    
    
"""

    FUNCIONES AUXILIARES EJERCICIO 1D

"""
# Doc: https://likegeeks.com/3d-plotting-in-python/

def kernel_2D(masc_gauss, masc_2deriv):
    # Creamos el producto de los kernels de las máscaras con la función outer y sumamos resultados
    Kernel2D_1 = np.outer(masc_gauss, masc_2deriv) 
    Kernel2D_2 = np.outer(masc_2deriv, masc_gauss) 
    Kernel2D = Kernel2D_1 + Kernel2D_2
    
    # doc: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    # https://stackoverflow.com/questions/54905088/how-to-graph-plot-2d-laplacian-of-gaussian-log-function-in-matlab-or-python
    x, y = Kernel2D.shape
    X,Y = np.meshgrid(np.arange(x),np.arange(y))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Modificar estos valores conlleva modificar "la camara", es decir, cambia la vista de la imagen
    ax.azim = -70
    ax.dist = 10
    ax.elev = 25
    
    # ax.plot_wireframe(X, Y, Kernel2D) -> solo se ven las lineas
    # https://www.analyticsvidhya.com/blog/2020/09/colormaps-matplotlib/
    ax.plot_surface(X, Y, Kernel2D, cmap= cm.twilight_shifted)

    # Quito las lineas de la figura porque así se ve más lindo
    ax.grid(False)
    plt.show()
    
    
def masc_laplaciana(imagen, v_sigmas):
    
    for i in v_sigmas:
        print("Calculamos y mostramos las máscaras para un sigma=" + str(i))
        print()
        masc_gauss = masc_gaussiana(funcion_gaussiana, None, i)
        masc_2deriv = masc_gaussiana(funcion_segunda_deriv_gaussiana, None, i)

        
        print("Mascara Gaussiana")
        print(masc_gauss)
        
        print()
        
        print("Mascara Segunda Derivada")
        print(masc_2deriv)
        
        titulo = "Mascaras Gaussiana y Segunda Derivada para sigma= " + str(i)
        plt.title(titulo) 
        plt.plot(masc_gauss)
        plt.plot(masc_2deriv)
        plt.legend(["Mascara Gaussiana", "Mascara Segunda Derivada"], loc= "lower right", fontsize = "x-small")
        plt.show()
        
        input("Pulse una tecla para continuar...")
        print()
        
        print("Mostramos el Kernel 2D equivalente")
        kernel_2D(masc_gauss, masc_2deriv)
        
            
        input("Pulse una tecla para continuar...")
        print()
        
        # convolucion2D(imagen, mascara2Der, mascaraGaussiana)
        conv1 = convolucion2D(imagen, masc_2deriv, masc_gauss)       
        
        # convolucion2D(imagen, mascaraGaussiana, mascara2D)
        conv2 = convolucion2D(imagen, masc_gauss,masc_2deriv)
        
        # Multiplicamos por sigma elevado al cuadrado para hacer una normalización de la escala
        laplaciana =  normaliza_imagen(i**2 * (conv1 + conv2))

        mostrarimagen(laplaciana, "Laplaciana de una Gaussiana")


def masc_laplacianaCV(imagen, v_sigmas):
   
    for i in v_sigmas:
        
        print("Laplaciana de la Gaussiana con OpenCV y sigma = " + str(i))
        
        _, k = calculo_tam_masc(i)

        # Usamos la funcion Laplacian de OpenCV
        # Doc: https://docs.opencv.org/3.4.15/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6
        # https://stackoverflow.com/questions/62589819/why-do-we-convert-laplacian-to-uint8-in-opencv
        # https://books.google.es/books?id=u3FGDwAAQBAJ&pg=PA150&lpg=PA150&dq=cv_8u+8-bit+unsigned+integer&source=bl&ots=z8EGXpnyqZ&sig=ACfU3U22aAgWuZB1ISNCiRjoP1YswBeCSg&hl=es&sa=X&ved=2ahUKEwiwid7ahvDzAhUCQEEAHXzXCxQQ6AF6BAgDEAM#v=onepage&q=cv_8u%208-bit%20unsigned%20integer&f=false
        
        laplaciana = cv2.Laplacian(imagen, cv2.CV_64F , ksize=k, borderType=cv2.BORDER_REFLECT)
        laplaciana =  normaliza_imagen(i**2 * laplaciana)
        
        mostrarimagen(laplaciana, "Laplaciana de una Gaussiana - OpenCV")
        
        input("Pulse una tecla para continuar...")
        print()
        
"""

    EJERCICIO 1D
    
"""  
   
def ejercicio1D(imagen, v_sigmas):

    masc_laplaciana(imagen, v_sigmas)
    
    print()
        
    print("Mostramos la Pirámide Gaussiana obtenida con OpenCv ")
    masc_laplacianaCV(imagen, v_sigmas)
        
    
    
    
       

"""

    FUNCIONES AUXILIARES EJERCICIO 2A

"""

def piramide_gaussiana(imagen, sigma, niveles):
    # Calculo de la máscara gaussiana
    masc_gauss = masc_gaussiana(funcion_gaussiana, None, sigma)
    
    piramide = []
    
    # Guardamos la imagen original como base de la pirámide
    piramide.append(imagen)
    
    for i in range(niveles):
        # Creamos un nuevo nivel alisando el anterior (padding)
        imagen_convolucionada = convolucion2D(piramide[-1], masc_gauss, masc_gauss)
        
        # Y quedándonos con la mitad de las filas y de las columnas
        imagen_convolucionada = imagen_convolucionada[::2, ::2]
        
        piramide.append(imagen_convolucionada)
        
    return np.array(piramide)


def piramide_gaussianaCV(imagen, niveles, borde):
    # Guardamos la imagen original como base de la pirámide
    piramide = []
    piramide.append(imagen)

    # para cada nivel, añadimos lo devuelto por pyrDown con el nivel anterior
    for i in range(niveles):
        piramide.append(cv2.pyrDown(piramide[-1], borderType=borde))

    return piramide

"""

    EJERCICIO 2A
    
"""      
    
def ejercicio2A(imagen, niveles, v_sigmas):
    
    for i in v_sigmas:
        
        piramide = piramide_gaussiana(imagen, i, niveles)
        
        print("Mostramos la Pirámide Gaussiana")
        print("Niveles " , niveles)
        print("Sigma " , i)
        
        pintaMI(piramide)
        
        input("Pulse una tecla para continuar...")
        
        print()
    
    # Comparamos nuestra solución con la función de OpenCV
    print("Mostramos la Pirámide Gaussiana obtenida con OpenCv")
    piramideCV = piramide_gaussianaCV(imagen, niveles, cv2.BORDER_REFLECT)
    pintaMI(piramideCV)
    


"""

    FUNCIONES AUXILIARES EJERCICIO 2B

"""

def piramide_laplaciana(imagen, sigma, niveles):
    # Calcular pirámide Gaussiana de la imagen
    piramide_gauss = piramide_gaussiana(imagen, sigma, niveles)
    
    piramide_laplaciana = []
    
    for i in range(niveles):

        # Poner el nivel i+1 de la piramide Gaussiana en el tamaño i (cv2.resize), interpolacion bilineal
        imagen_expandida = piramide_gauss[i + 1]
        
        # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
        height, width = piramide_gauss[i].shape
        dim = (int(width), int(height))
        imagen_expandida = cv2.resize(imagen_expandida, dsize=dim)
        
        # Calcular piramide_gauss[i] - imagen_expandida
        imagen_laplaciana = piramide_gauss[i] - imagen_expandida
        
        piramide_laplaciana.append(imagen_laplaciana)
       
    # Guardamos el ultimo nivel de la piramide Gaussiana para poder reconstruir la imagen original
    piramide_laplaciana.append(piramide_gauss[-1])
    
    return np.array(piramide_laplaciana)


def piramide_laplacianaCV(imagen, niveles, borde):

    # calculamos la piramide gaussiana utilizando opencv
    piramide_gauss = piramide_gaussianaCV(imagen, niveles, borde)

    piramide_laplaciana = []

    for i in range(niveles):
        # usamos pyrUp para reescalar el nivel anterior
        dim = (piramide_gauss[i].shape[1], piramide_gauss[i].shape[0])
        # Poner el nivel i+1 de la piramide Gaussiana en el tamaño i
        imagen_expandida = cv2.pyrUp(piramide_gauss[i+1], dstsize=dim)

        # la restamos con el nivel anterior
        imagen_laplaciana = piramide_gauss[i] - imagen_expandida

        piramide_laplaciana.append(imagen_laplaciana)
        
    # Guardamos el ultimo nivel de la piramide Gaussiana para poder reconstruir la imagen original
    piramide_laplaciana.append(piramide_gauss[-1])

    return piramide_laplaciana


"""

    EJERCICIO 2B
    
"""  
   
def ejercicio2B(imagen, niveles, sigma):

    piramide = piramide_laplaciana(imagen, sigma, niveles)
    
    print("Mostramos la Pirámide Laplaciana")
    print("Niveles " , niveles)
    print("Sigma " , sigma)
    
    pintaMI(piramide)
    
    input("Pulse una tecla para continuar...")
    
    print()
    
    # Comparamos nuestra solución con la función de OpenCV
    print("Mostramos la Pirámide Laplaciana obtenida con OpenCv")
    piramideCV = piramide_laplacianaCV(imagen, niveles, cv2.BORDER_REFLECT)
    pintaMI(piramideCV)
    
    return piramide



"""

    FUNCIONES AUXILIARES EJERCICIO 2C

"""

def reconstruir_imagen(piramide_laplaciana):
    vim = []
    niveles = []
    
    # Partimos del último nivel de la piramide Laplacina de la imagen 
    imagen_reconstruida = piramide_laplaciana[-1]
    
    # Recorremos la piramide desde el nivel mas alto aplicando el algoritmo
    # https://www.kite.com/python/answers/how-to-iterate-through-a-decreasing-range-with-a-for-loop-in-python
    for i in range(len(piramide_laplaciana)-1, 0, -1): 
        # La idea es ampliar y sumar los detalles
        # Expandir el nivel al tamaño del nivel anterior (cv2.resize) usando interpolacion bilineal
        height, width = piramide_laplaciana[i-1].shape
        dim = (int(width), int(height))
        imagen_expandida = cv2.resize(imagen_reconstruida, dsize=dim)

        # Calculas piramide_laplaciana[i] + imagen_expandida
        imagen_reconstruida = piramide_laplaciana[i-1] + imagen_expandida
        
        # Para mostrar el proceso de reconstrucción en un mismo plot
        
        imagen_normalizada = normaliza_imagen(imagen_reconstruida)
        vim.append(imagen_normalizada)
        titulo = "Nivel " + str(i)
        niveles.append(titulo)
       
        mostrarimagen(imagen_normalizada)
    
    pintaConTitulo(vim, niveles)
    
    return imagen_reconstruida
        

def calcula_error_euclideo(imagen, imagen_reconstruida):
    
    #return sum((sum(imagen[:,:] - imagen_reconstruida[:,:]))**2)
    matriz_diferencias = np.sqrt(sum((imagen[:] - imagen_reconstruida[:])**2))
       
    return sum(matriz_diferencias)
                  
"""

EJERCICIO 2C
    
"""  
   
def ejercicio2C(imagen, piramide_laplaciana):
    imagen_reconstruida = reconstruir_imagen(piramide_laplaciana)
    
    input("Pulse una tecla para continuar...")
    
    print()
    print("Calculamos el error entre ambas imagenes mediante la distancia Euclídea")
    
    error = calcula_error_euclideo(imagen, imagen_reconstruida)
    
    print("El error entre las dos imágenes es: ", error)
    
    
 
    
 
    
"""

    FUNCIONES AUXILIARES EJERCICIO BONUS 1

"""


def imagen_hibrida(img1, img2, sigma_baja, sigma_alta):
     # Calculamos la máscara para la imagen de bajas frecuencias
    hmask = masc_gaussiana(funcion_gaussiana, None, sigma_baja)
    vmask = np.copy(hmask)
    
    # Obtenemos la imagen de bajas frecuencias aplicando el filtro Gaussiano para alisar fuertemente
     # Imagen monobanda
    if len(img1.shape) == 2:
        img_freq_bajas = convolucion2D(img1, hmask, vmask)
    else:
        # Realizamos la convolucion para cada uno de los canales BGR.
        # De esta manera tenemos tres matrices 2D representando los tres canales de color
        # El resultado será la concatenacion de las imagenes, y para ello antes tenemos que
        # añadir un nuevo eje, ya que nuestro resultado es un array 3D!!
        # Doc: https://wjngkoh.wordpress.com/2015/01/29/numpy-concatenate-three-images-rgb-into-one/
        # https://qastack.mx/programming/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
        b = convolucion2D(img1[:,:,0], hmask, vmask)[:,:,np.newaxis]
        g = convolucion2D(img1[:,:,1], hmask, vmask)[:,:,np.newaxis]
        r = convolucion2D(img1[:,:,2], hmask, vmask)[:,:,np.newaxis]
    
        #img_freq_bajas = img1.merge("RGB",(b,g,r))
        img_freq_bajas =  np.concatenate((b, g, r), axis=2)
        
    img_freq_bajas = normaliza_imagen(img_freq_bajas)
    
    # Calculamos la máscara para la imagen de altas frecuencias
    hmask = masc_gaussiana(funcion_gaussiana, None, sigma_alta)
    vmask = np.copy(hmask)
    
    if len(img1.shape) == 2:
        img_alisada = convolucion2D(img2, hmask, vmask)
    else:
        b = convolucion2D(img2[:,:,0], hmask, vmask)[:,:,np.newaxis]
        g = convolucion2D(img2[:,:,1], hmask, vmask)[:,:,np.newaxis]
        r = convolucion2D(img2[:,:,2], hmask, vmask)[:,:,np.newaxis]

        img_alisada =  np.concatenate((b, g, r), axis=2)
        
    
    img_freq_altas = img2 - img_alisada
    img_freq_altas = normaliza_imagen(img_freq_altas)
    
    img_hibrida = img_freq_bajas + img_freq_altas
    img_hibrida = normaliza_imagen(img_hibrida)
    
    return img_freq_bajas, img_freq_altas, img_hibrida  



"""

EJERCICIO BONUS 1

"""

def ejercicio_Bonus_1(imagen_baja, imagen_alta, sigmas_bajos, sigmas_altos):
    
    for sigma_bajo, sigma_alto in zip(sigmas_bajos, sigmas_altos):
        print("SIGMA DE BAJAS FRECUENCIAS: " + str(sigma_bajo) + " , SIGMA DE ALTAS FRECUENCIAS " + str(sigma_alto))
        print()
        
        img_freq_bajas, img_freq_altas, img_hibrida = imagen_hibrida(imagen_baja, imagen_alta, sigma_bajo, sigma_alto)
        print("Imagen de frecuencias bajas: ")
        mostrarimagen(img_freq_bajas, "Imagen de frecuencias bajas")
        
        print("Imagen de frecuencias altas: ")
        mostrarimagen(img_freq_altas, "Imagen de frecuencias altas")
        
        print("Imagen de frecuencias híbrida: ")
        mostrarimagen(img_hibrida, "Imagen híbrida")
        
        print()



"""

    MAIN DEL PROGRAMA
    
"""



print("----------- EJERCICIO 1 -----------")    
print(" APARTADO A ")

print()

ejercicio1A()


print()
print(" APARTADO B ")

print()

ejercicio1B()

input("Pulse una tecla para continuar...")

print()
print(" APARTADO C ")

print()

# Leemos la imagen
filename = './imagenes/einstein.bmp'
img = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
img_float = img_a_float(img)

print("IMAGEN ORIGINAL: ")
mostrarimagen(img, "Imagen original")

print()

# Creamos una lista con distintos sigmas para luego iterar sobre ella
v_sigmas = [1,3,5]
    
ejercicio1C(img_float, v_sigmas)

print(" APARTADO C-A ")

print()

print("Realizamos la comparación con GaussianBlur: ")
print()
ejercicio1C_A(img, v_sigmas)

print(" APARTADO C-B ")

print()

print("Calculamos las imagenes derivadas respecto de x y de y de una imagen dada:")
print()

ejercicio1C_B(img, v_sigmas)

print()
print(" APARTADO D ")

v_sigmas = [1,3]

ejercicio1D(img_float, v_sigmas)

print()



print("----------- EJERCICIO 2 -----------")    

niveles = 4
v_sigmas = [1,2,3]

print(" APARTADO A ")

print()

ejercicio2A(img_float, niveles, v_sigmas)

input("Pulse una tecla para continuar...")

print()

print(" APARTADO B ")

print()
niveles = 4
sigma = 1
piramide = ejercicio2B(img_float, niveles, sigma)

input("Pulse una tecla para continuar...")

print()


print(" APARTADO C ")

print()

print("Reconstrucción de la imagen original a partir de la pirámide Laplaciana")
ejercicio2C(img, piramide)

input("Pulse una tecla para continuar...")
print()

print("----------- BONUS -----------")    
print(" APARTADO 1 ")
print()

print("EXPERIMENTAMOS CON LAS IMÁGENES DE EINSTEIN Y MARILYN: ")
print()

filename = './imagenes/einstein.bmp'
einstein = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
einstein_float = img_a_float(einstein)


filename = './imagenes/marilyn.bmp'
marilyn = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
marilyn_float = img_a_float(marilyn)

# Experimentar con distintos sigmas
sigmas_bajos = [2,4,5,7]
sigmas_altos = [1,2,3,5]

ejercicio_Bonus_1(einstein_float, marilyn_float, sigmas_bajos, sigmas_altos)

input("Pulse una tecla para continuar...")

print("Mostramos la pirámide Gaussiana con los sigmas que mejor se han adecuado a la imagen: ")
sigma_bajo = 5
sigma_alto = 3

print("Elección: sigma bajo de " + str(sigma_bajo) + " y sigma alto de " + str(sigma_alto))

a, b, res = imagen_hibrida(einstein_float, marilyn_float, sigma_bajo, sigma_alto)

piramide = piramide_gaussiana(res, 1, 4)
pintaMI(piramide)

print()
input("Pulse una tecla para continuar...")

################################################################################
################################################################################

print("EXPERIMENTAMOS CON LAS IMÁGENES DEL PÁJARO Y EL AVIÓN: ")
print()

filename = './imagenes/bird.bmp'
bird = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
bird_float = img_a_float(bird)


filename = './imagenes/plane.bmp'
plane = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
plane_float = img_a_float(plane)

# Experimentar con distintos sigmas
sigmas_bajos = [2,5,7,10]
sigmas_altos = [1,3,4,4]

ejercicio_Bonus_1(bird_float, plane_float, sigmas_bajos, sigmas_altos)

input("Pulse una tecla para continuar...")

print("Mostramos la pirámide Gaussiana con los sigmas que mejor se han adecuado a la imagen: ")
sigma_bajo = 7
sigma_alto = 3

print("Elección: sigma bajo de " + str(sigma_bajo) + " y sigma alto de " + str(sigma_alto))

a, b, res1 = imagen_hibrida(bird_float, plane_float, sigma_bajo, sigma_alto)
piramide = piramide_gaussiana(res1, 1, 4)
pintaMI(piramide)

print()
input("Pulse una tecla para continuar...")

################################################################################
################################################################################

print("EXPERIMENTAMOS CON LAS IMÁGENES DEL PEZ Y EL SUBMARINO: ")
print()

filename = './imagenes/fish.bmp'
fish = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
fish_float = img_a_float(fish)


filename = './imagenes/submarine.bmp'
submarine = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
submarine_float = img_a_float(submarine)

# Experimentar con distintos sigmas
sigmas_bajos = [2,3,5,5]
sigmas_altos = [1,2,3,4]

ejercicio_Bonus_1(fish, submarine, sigmas_bajos, sigmas_altos)

input("Pulse una tecla para continuar...")

print("Mostramos la pirámide Gaussiana con los sigmas que mejor se han adecuado a la imagen: ")
sigma_bajo = 5
sigma_alto = 4

print("Elección: sigma bajo de " + str(sigma_bajo) + " y sigma alto de " + str(sigma_alto))

a, b, res2 = imagen_hibrida(fish_float, submarine_float, sigma_bajo, sigma_alto)

piramide = piramide_gaussiana(res2, 1, 4)
pintaMI(piramide)

input("Pulse una tecla para continuar...")
print()



print(" APARTADO 2 ")
print()

filename = './imagenes/einstein.bmp'
einstein = leeimagen(filename, cv2.IMREAD_COLOR)


filename = './imagenes/marilyn.bmp'
marilyn = leeimagen(filename, cv2.IMREAD_COLOR)
marilyn_float = img_a_float(marilyn)

sigma_bajo = 5
sigma_alto = 3

img_freq_bajas, img_freq_altas, img_hibrida = imagen_hibrida(einstein, marilyn, sigma_bajo, sigma_alto)

print("Imagen de frecuencias bajas: ")
mostrarimagen(img_freq_bajas, "Imagen de frecuencias bajas")

print("Imagen de frecuencias altas: ")
mostrarimagen(img_freq_altas, "Imagen de frecuencias altas")

print("Imagen de frecuencias híbrida: ")
mostrarimagen(img_hibrida, "Imagen híbrida")

filename = './imagenes/bird.bmp'
einstein = leeimagen(filename, cv2.IMREAD_COLOR)


filename = './imagenes/plane.bmp'
marilyn = leeimagen(filename, cv2.IMREAD_COLOR)
marilyn_float = img_a_float(marilyn)

sigma_bajo = 7
sigma_alto = 3

img_freq_bajas, img_freq_altas, img_hibrida = imagen_hibrida(einstein, marilyn, sigma_bajo, sigma_alto)

print("Imagen de frecuencias bajas: ")
mostrarimagen(img_freq_bajas, "Imagen de frecuencias bajas")

print("Imagen de frecuencias altas: ")
mostrarimagen(img_freq_altas, "Imagen de frecuencias altas")

print("Imagen de frecuencias híbrida: ")
mostrarimagen(img_hibrida, "Image híbrida")

filename = './imagenes/fish.bmp'
einstein = leeimagen(filename, cv2.IMREAD_COLOR)


filename = './imagenes/submarine.bmp'
marilyn = leeimagen(filename, cv2.IMREAD_COLOR)
marilyn_float = img_a_float(marilyn)

sigma_bajo = 5
sigma_alto = 4

img_freq_bajas, img_freq_altas, img_hibrida = imagen_hibrida(einstein, marilyn, sigma_bajo, sigma_alto)

print("Imagen de frecuencias bajas: ")
mostrarimagen(img_freq_bajas, "Imagen de frecuencias bajas")

print("Imagen de frecuencias altas: ")
mostrarimagen(img_freq_altas, "Imagen de frecuencias altas")

print("Imagen de frecuencias híbrida: ")
mostrarimagen(img_hibrida, "Imagen híbrida")


input("Pulse una tecla para continuar...")
print()

print(" APARTADO 3 ")
print()

filename = './imagenes/dalai.jpg'
dalai = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
dalai_float = img_a_float(dalai)

filename = './imagenes/llama.jpg'
llama = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
llama_float = img_a_float(llama)

height_dalai, width_dalai = dalai_float.shape

height_llama, width_llama = llama_float.shape

if height_llama <= height_dalai:
    h_min = height_llama
else:
    h_min = height_dalai
    
if width_llama <= width_dalai:
    w_min = width_llama
else:
    w_min = width_dalai

dim = (w_min, h_min)

llama_resize = cv2.resize(llama_float, dsize=dim)
dalai_resize = cv2.resize(dalai_float, dsize=dim)

sigma_bajo = 7
sigma_alto = 5

img_freq_bajas, img_freq_altas, img_hibrida = imagen_hibrida(llama_resize, dalai_resize, sigma_bajo, sigma_alto)

print("Imagen de frecuencias bajas: ")
mostrarimagen(img_freq_bajas, "Imagen de frecuencias bajas")

print("Imagen de frecuencias altas: ")
mostrarimagen(img_freq_altas, "Imagen de frecuencias altas")

print("Imagen de frecuencias híbrida: ")
mostrarimagen(img_hibrida, "Imagen híbrida")




filename = './imagenes/dalai.jpg'
dalai = leeimagen(filename, cv2.IMREAD_COLOR)
dalai_float = img_a_float(dalai)


filename = './imagenes/llama.jpg'
llama = leeimagen(filename, cv2.IMREAD_COLOR)
llama_float = img_a_float(llama)

height_dalai, width_dalai, depth_dalai = dalai_float.shape

height_llama, width_llama, depth_llama = llama_float.shape

if height_llama <= height_dalai:
    h_min = height_llama
else:
    h_min = height_dalai
    
if width_llama <= width_dalai:
    w_min = width_llama
else:
    w_min = width_dalai

dim = (w_min, h_min)

llama_resize = cv2.resize(llama_float, dsize=dim)
dalai_resize = cv2.resize(dalai_float, dsize=dim)

sigma_bajo = 7
sigma_alto = 5

img_freq_bajas, img_freq_altas, img_hibrida = imagen_hibrida(llama_resize, dalai_resize, sigma_bajo, sigma_alto)

print("Imagen de frecuencias bajas: ")
mostrarimagen(img_freq_bajas, "Imagen de frecuencias bajas")

print("Imagen de frecuencias altas: ")
mostrarimagen(img_freq_altas, "Imagen de frecuencias altas")

print("Imagen de frecuencias híbrida: ")
mostrarimagen(img_hibrida, "Imagen híbrida")
