# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 12:38:25 2021

@author: Alba Casillas Rodríguez
"""

"""
Se importan las libererías necesarias para la práctica:
    
"""

import numpy as np
import cv2
import math
import random


""" 

    FUNCIONES AUXILIARES

"""

# Función que lee una imagen a partir de un fichero. En esta práctica leeremos las imagenes en escala de grises (flagColor)

def leeimagen(filename, flagColor):
    # Cargamos la imagen
    im = cv2.imread(filename,flagColor) 
    
    return im

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
    

# Funcion utilizada para saber por que componente del vector ordenar una lista

def SortValue(lista_vector):
    return lista_vector[0]

"""

    FUNCIONES DE LA PRACTICA 1 PARA CALCULAR LA CONVOLUCION

"""

def calculo_k(sigma):
    k = int((3*sigma))
    
    return k

def funcion_gaussiana(x, sigma):
    
    return math.exp(- (x**2) / (2 * (sigma**2)))

def masc_gaussiana(funcion, sigma):
    
    masc = []
    
    k = calculo_k(sigma)

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
    
    return masc_norm


def convolucion2D(imagen, hmask, vmask):
    # Guardamos el alto y ancho de la imagen original
    # el bucle tiene que dar tantas vueltas como filas tenga la imagen original, sin el padding
    height_img, width_img = imagen.shape
    
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

    # Guardamos el alto y ancho de la imagen original
    # el bucle tiene que dar tantas vueltas como filas tenga la imagen original, sin el padding
    height_matrix, width_matrix = matrix_tmp_traspuesta.shape

    
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

    FUNCIONES EJERCICIO 1

"""

# Funcion que calcula el sigma_k (absoluto)

def sigma_absoluto(sigma_ad, k, ns):
    return sigma_ad * (2**(k/ns))

# Realizamos el calculo de sigma_s con la formula vista en las practicas

def sigma_suavizado(sigma_0,s, ns):
    return (sigma_0 * (math.sqrt(2**(2*s/ns) - 2**(2*(s-1)/ns))))


# Función que calcula los sigmas absoluto y suavizado necesarios.

def calculo_sigmas(sigma_0, sigma_ad,  ns, num_octavas):
    lista_sigmas = []
    listas_sigmas_abs = []
    
    # Para este primer bucle, calculamos la lista de sigmas con las que se calculan
    # los incrementos, es decir, los sigmas con los que se convolucionará la imagen
    # en cada escala.
    # Usamos ns = 3 + 2 escalas extra (en el bucle se suma +3 al ser una cota abierta 
    # por la derecha). Estas escalas extra se usarán para el cálculo de vecinos.
    
    for i in range(1, ns+3):
        sigma = sigma_suavizado(sigma_0, i, 3)
        
        lista_sigmas.append(sigma)
    
    # Este bucle calcula el valor de los sigmas real/absoluto. Por ello se calcula
    # un sigma por cada escala de cada una de las octavas, donde se añade también
    # el sigma de adquisición (0.8)
    
    num_iteraciones = num_octavas * ns + 1
    listas_sigmas_abs.append(sigma_ad)
    
    for k in range(num_iteraciones):
        sigma_abs = sigma_absoluto(sigma_ad, k, ns)
        listas_sigmas_abs.append(sigma_abs)
        
    
    return lista_sigmas, listas_sigmas_abs
        

# Función que calcula todas las escalas de una sola octava. Para ello usamos los
# sigma de suavizado calculados y convolucionamos la imagen en cada escala.
        
def calculo_escalas(imagen, lista_sigmas):
    escala_imgs = []
    
    # Se empieza en la escala 1 y realmente es hasta ns+2, lo que pasa es que para 
    # calcular las 5 escalas hay que hacer ns+3 porque el bucle es < no estricto.
    imgConvolucionada = imagen
    escala_imgs.append(imgConvolucionada)
    
    for i in lista_sigmas: 
        hmask = masc_gaussiana(funcion_gaussiana, i)
        vmask = np.copy(hmask)
       
        imgConvolucionada = convolucion2D(imgConvolucionada, hmask, vmask)
        
        escala_imgs.append(imgConvolucionada)
           
    return escala_imgs
    
    
# Función que calcula las diferencias de Gaussianas (DoG) entre las escalas j y j+1
# de la octava i.

def piramide_lowe(octavas):
    DoG_octavas = []
    
    # Para cada una de las octavas 
    for i in range(len(octavas)):
        
        print("Octava " + str(i))
        print()
        
        DoG = []
        
        # Para cada una de sus escals
        for j in range(len(octavas[i])-1):

            # Se realiza las diferencias de Gaussianas
            diferencia_gauss = octavas[i][j] - octavas[i][j+1]

            # La diferencia de Gauss es entre j y j+1 pero cambiamos el print para que las escalas vayan de [1-5]
            print("DoG entre escalas " + str(j+1) + " y " + str(j+2))
            print()
            normalizada = normaliza_imagen(diferencia_gauss)
            titulo = "Diferencias de Gaussianas en octava " + str(i)
            mostrarimagen(normalizada, titulo)
                   
            DoG.append(diferencia_gauss)
            
        input("Pulse una tecla para continuar...")
        print()
            
    
        DoG_octavas.append(DoG)
        
    return DoG_octavas

# Funcion que calcula el calculo de vecinos entre tres escalas consecutivas y obtiene los extremos locales
def calculo_vecinos(esc_arriba, esc_central, esc_abajo, sigma_k, delta, escala):
    height, width = np.array(esc_central).shape
    
    radio = 1 # Queremos obtener los vecinos adyacentes (el de arriba, al lado, abajo....)
    vector_extremos = []
    
    # Recorremos la escala central
    for row in range(1, height-1):
        for column in range(1, width-1):
            # Guardamos el valor y coordenadas del pixel que estoy evaluando
            mi_valor = esc_central[row][column]
            mis_coord = (row, column)
            
            # Obtenemos el vecindario creando submatrices de dimension 3x3x3 que engloble el pixel evaluado
            # y sus vecinos adyacentes (por eso se suma/resta el radio, que marca el numero de vecinos a obtener).
            mis_vecinos_central = esc_central[(row-radio):(row+radio+1),(column-radio):(column+radio+1)]
            mis_vecinos_arriba = esc_arriba[(row-radio):(row+radio+1),(column-radio):(column+radio+1)]
            mis_vecinos_abajo = esc_abajo[(row-radio):(row+radio+1),(column-radio):(column+radio+1)]

            # Se obtiene el maximo y el minimo de cada escala
            # https://thispointer.com/find-max-value-its-index-in-numpy-array-numpy-amax/
            max_central = np.amax(mis_vecinos_central)
            min_central = np.amin(mis_vecinos_central) 

            
            max_arriba = np.amax(mis_vecinos_arriba)
            min_arriba = np.amin(mis_vecinos_arriba) 
          
            max_abajo = np.amax(mis_vecinos_abajo)
            min_abajo = np.amin(mis_vecinos_abajo) 


            # Se calcula cual es el maximo de los maximos y el minimo de los minimos
            vecinos_max = [max_arriba, max_central, max_abajo]
            el_maximo = max(vecinos_max)
            
            vecinos_min = [min_arriba, min_central, min_abajo]
            el_minimo = min(vecinos_min)
            
            
            # Si el pixel que estoy evaluando corresponde a un maximo o minimo, significa que es extremo local y se guarda
            if mi_valor == el_maximo or mi_valor == el_minimo:
                # Guardamos obligatoriamente (x,y, sigma_k) pero tambien el valor y el delta para operaciones posteriores
                soy_extremo = [abs(mi_valor), mis_coord[0], mis_coord[1],sigma_k, delta, escala]
               
                vector_extremos.append(soy_extremo)

    
    return vector_extremos
                        

def ejercicio_1(imagen, sigma_ad):
    num_octavas = 4
    octavas = []
    ns = 3
    delta_0 = 0.5
    k = 3 # Iterador para usar el sigma correcto en el calculo de los extremos
    
    
    # Obtengo las dimensiones de la imagen para la interpolacion bilineal
    height, width = imagen.shape
    dim = (width*2, height*2) # *2 porque la duplicamos

    # Duplicamos el tamanio de la imagen y, por tanto, el espacio cubierto
    # por la operacion de suavizado se duplica
    imagen_dup = cv2.resize(imagen, dsize=dim)
    
    # el sigma de 0.8 pasa a ser 1.6
    sigma_ini = sigma_ad / delta_0
    
    # Calculamos los sigmas con los que vamos a calcular las escalas
    # Los sigmas no varian entre el calculo de una octava y otra
    lista_sigmas, listas_sigmas_abs = calculo_sigmas(sigma_ini, sigma_ad, ns, num_octavas)
    
    print("Se calculan las escalas de todas las octavas")
    print()
    
    # Calculamos la primera octava con la imagen duplicada
    imagenes_reescaladas = calculo_escalas(imagen_dup, lista_sigmas)
    octavas.append(imagenes_reescaladas)
    
    # Contando con que en el vector las imagenes se guardan desde la pos 0-4
    # la imagen de la escala 3 estara almacenada en la posicion 2
    pos_elegida = 3
    img_elegida = imagenes_reescaladas[pos_elegida-1]

    # Para el calculo de una octava i, tendremos que coger la escala 3 de la octava i-1
    # y reducirla por interpolacion bilineal. Esta imagen correspondera a la escala 0
    # octava i y se podran calcular el resto de las octavas.
    for i in range(2, num_octavas+1):
        height, width = img_elegida.shape
        dim = (math.ceil(width/2), math.ceil(height/2)) # /2 porque la muestreamos a la mitad
        img_muestreada = cv2.resize(img_elegida, dsize=dim)
        
        imagenes_reescaladas = calculo_escalas(img_muestreada, lista_sigmas)
        img_elegida = imagenes_reescaladas[pos_elegida-1]
        octavas.append(imagenes_reescaladas)
        
        imagenes_reescaladas = []
        
    
    # Mostramos para cada una de las octavas
    for i in range(len(octavas)):
        print("Octava " + str(i))
        
        # Mostramos todas las escalas de cada octava
        for j in range(ns):
            normalizada = normaliza_imagen(octavas[i][j])
    
            titulo = "Octava " + str(i) + " escala " + str(j+1)
            mostrarimagen(normalizada, titulo)
        
        input("Pulse una tecla para continuar...")
        print()
     
    
    print("Se calculan las diferencias gaussianas")
    DoG_octavas = piramide_lowe(octavas)

    lista_extremos = []
    los_mejores = []
    extrm_oct1 = []
    extrm_oct2 = []
    extrm_oct3 = []
    extrm_oct4 = []
    

    print("Identificamos los 100 extremos locales de mayor respuesta")
    delta = 0.5
    
    # Se realiza el calculo de los extremos locales, los cuales se irán guardando en lista_extremos
    for i in range(len(DoG_octavas)):
        escala = 3
        extremos = calculo_vecinos(DoG_octavas[i][4], DoG_octavas[i][3], DoG_octavas[i][2], listas_sigmas_abs[k+1], delta, escala)
        lista_extremos.extend(extremos)
        
        # Calculo de vecinos de las Diferencias de Gaussianas (3-2-1)
        escala = 2 # Escala de la DOG de donde cogemos el sigma_k
        extremos1 = calculo_vecinos(DoG_octavas[i][3], DoG_octavas[i][2], DoG_octavas[i][1], listas_sigmas_abs[k], delta, escala)
        lista_extremos.extend(extremos1)

        # Calculo de vecinos de las Diferencias de Gaussianas (2-1-0)
        escala = 1 # Escala de la DOG de donde cogemos el sigma_k
        extremos2 = calculo_vecinos(DoG_octavas[i][2], DoG_octavas[i][1], DoG_octavas[i][0], listas_sigmas_abs[k-1], delta, escala)
        lista_extremos.extend(extremos2)
        
        
        # ESTA PARTE DEL CÓDIGO ES DEL BONUS DEL EJERCICIO 1
        # Se guardan en listas diferentes los extremos encontrado en cada octava
        # la cual la distinguimos por su delta correspondiente
        # Los extremos se ordenan de mayor a menor respuesta y se elige el porcentaje especificado en el enunciado
        # Octava 1: 50% del total
        # Octava 2: 25% del total
        # Octava 3: 15% del total
        # Octava 4: 10% del total
        if delta == 0.5:
            extrm_oct1.extend(extremos)
            extrm_oct1.extend(extremos2)
            
            extrm_oct1.sort(key=SortValue, reverse=True)
            top_oct1_50 = extrm_oct1[:50]
            los_mejores.extend(top_oct1_50)
            
        elif delta == 1:
            extrm_oct2.extend(extremos)
            extrm_oct2.extend(extremos2)
            
            extrm_oct2.sort(key=SortValue, reverse=True)
            top_oct2_25 = extrm_oct2[:25]
            los_mejores.extend(top_oct2_25)
            
        elif delta == 2:
            extrm_oct3.extend(extremos)
            extrm_oct3.extend(extremos2)
            
            extrm_oct3.sort(key=SortValue, reverse=True)
            top_oct3_15 = extrm_oct3[:15]
            los_mejores.extend(top_oct3_15)
            
        elif delta == 4:
            extrm_oct4.extend(extremos)
            extrm_oct4.extend(extremos2)
            
            extrm_oct4.sort(key=SortValue, reverse=True)
            top_oct4_10 = extrm_oct4[:10]
            los_mejores.extend(top_oct4_10)
        
        k = k + ns # Actualizamos K para el calculo del sigma absoluto
        delta = delta * 2 # Actualizamos delta para saber en que octava estamos

        

    # Ordenamos la lista de extremos de mayor a menor respuesta (valor del pixel)
    lista_extremos.sort(key=SortValue, reverse=True)
    
    # Nos quedamos con los 100 extremos de mayor respuesta
    top_100 = np.copy(lista_extremos[:100])
    
    keypoints = []
    keypoints_octavas = []
    radio = 6
    top = np.copy(top_100) # creamos la copia porque top_100 se modificara...
    
    # Convertimos los extremos en Keypoints con cv2.KeyPoint
    for i in range(len(top_100)):
        # Escalamos las coordenadas (parametros 1 y 2) a las de la imagen original multiplicando por su delta.
        top_100[i][1] = top_100[i][1] * top_100[i][4]
        top_100[i][2] = top_100[i][2] * top_100[i][4]
        
        # https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html#a9d81b57ae182dcb3ceac86a6b0211e94
        # https://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column
        keypoints.append(cv2.KeyPoint(top_100[i][2], top_100[i][1], size =(top_100[i][3]*radio*2)))
        
    # https://docs.opencv.org/3.4/d4/d5d/group__features2d__draw.html#gab958f8900dd10f14316521c149a60433
    # https://www.programcreek.com/python/example/89309/cv2.drawKeypoints
    imagen_resultado = cv2.drawKeypoints(imagen, keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    mostrarimagen(imagen_resultado)
    
    input("Pulse una tecla para continuar...")
    print()
    
    print("Identificamos los extremos locales de mayor respuesta de cada octava")

    
    # Convertimos los extremos en Keypoints con cv2.KeyPoint
    for i in range(len(los_mejores)):
        los_mejores[i][1] = los_mejores[i][1] * los_mejores[i][4]
        los_mejores[i][2] = los_mejores[i][2] * los_mejores[i][4]
        
        keypoints_octavas.append(cv2.KeyPoint(los_mejores[i][2], los_mejores[i][1], size =((los_mejores[i][3]*radio)*2)))
        
    
    imagen_resultado = cv2.drawKeypoints(imagen, keypoints_octavas, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    mostrarimagen(imagen_resultado)

        
    return DoG_octavas, top

    
  
    
    
"""

    FUNCIONES EJERCICIO 2

"""    
 
def obtener_descriptor(imagen):
    # Creamos un objeto SIFT del que extraer las caracteristicas
    sift = cv2.SIFT_create()
    
    # Se extraen los keypoints y el descriptor de la imagen
    keypoints, descriptor = sift.detectAndCompute(imagen, None)
    
    return keypoints, descriptor
    
def matchs_brute_force(img1, kpts1, desc1, img2, kpts2, desc2):
    # Establecemos los puntos de correspondencia entre las 2 imagenes 
    # para ello usaremos BFMatcher
    # Segun la documentacion, los normType preferibles para el descriptor SIFT son NORM_L1 y NORM_L2
    # Buscando que diferencias hay entre ellas, Norm L1 usa la distancia Manhattan 
    # y Norm L2 la distancia euclidea, recomendada en clase.
    # Doc: https://docs.opencv.org/4.5.3/d3/da1/classcv_1_1BFMatcher.html
    # Doc: http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
    # Doc: https://stackoverflow.com/questions/32849465/difference-between-cv2-norm-l2-and-cv2-norm-l1-in-opencv-python  
    
    # BruteForce + CrossCheck
    # BFMatcher, como su nombre indica, establece fuerza bruta
    # Establecemos el parametro crossCheck a True; de forma que el matcher devuelve solo aquellos matches(i,j)
    # que se elijan en ambos sentidos (entre ellos)
    matcher1 = cv2.BFMatcher_create(cv2.NORM_L2,crossCheck=True)
    
    # Obtenemos los matches
    # Usamos el metodo match para realizar la correspondencia a raiz de los dos descriptores
    mchts1 = matcher1.match(desc1,desc2)
    random_mchts1 = random.sample(mchts1, 100)
    
    matched_image = cv2.drawMatches(img1, kpts1, img2, kpts2, random_mchts1, None, flags=2)
    mostrarimagen(matched_image)
    
def matchs_lowe_average2nn(img1, kpts1, desc1, img2, kpts2, desc2):
    # Establecemos los puntos de correspondencia entre las 2 imagenes 
    # para ello usaremos BFMatcher
    # Segun la documentacion, los normType preferibles para el descriptor SIFT son NORM_L1 y NORM_L2
    # Buscando que diferencias hay entre ellas, Norm L1 usa la distancia Manhattan 
    # y Norm L2 la distancia euclidea, recomendada en clase.
    # Doc: https://docs.opencv.org/4.5.3/d3/da1/classcv_1_1BFMatcher.html
    # Doc: http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
    # Doc: https://stackoverflow.com/questions/32849465/difference-between-cv2-norm-l2-and-cv2-norm-l1-in-opencv-python  
    
    # Lowe-Average-2NN
    # Usamos la misma funcion para crear el matcher. 
    # Como crossCheck es False, el comportamiento de BFMatcher sera encontrar el K vecino mas cercano por cada query descriptor
    # Por tanto, debemos usar la funcion knnMatch con k=2 (porque en el enunciado dice que es 2NN)
    matcher2 = cv2.BFMatcher_create(cv2.NORM_L2)
    
    # Obtenemos los matches
    mchts2 = matcher2.knnMatch(desc1,desc2,k=2)
    
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # Corregimos el error de:
    #   > Overload resolution failed:
    #   >  - Expected cv::DMatch for argument 'matches1to2'
    #   >  - Expected cv::DMatch for argument 'matches1to2'
    #   >  - drawMatches() missing required argument 'matchesThickness' (pos 7)
    #   >  - drawMatches() missing required argument 'matchesThickness' (pos 7)
    #
    # Este fallo ocurre porque para este caso debemos de aplicar el criterio de Lowe para 
    # saber si se establece un match o no!
           
    matchesMask = []
    
    for m1,m2 in mchts2:
        # David Lowe propone que para establecer si hay match o no el criterio de:
        # m es mejor match si y solamente si d(m1) < 0.8*d(m2), siendo d la distancia
        if m1.distance < 0.8*m2.distance:
            matchesMask.append([m1])
            
    random_mchts2 = random.sample(matchesMask, 100)
    
    matched_image = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, random_mchts2, None, flags=2)
    mostrarimagen(matched_image)
    
 
def correspondencias(img1, img2):
    
    kpts1, desc1 = obtener_descriptor(img1)
    kpts2, desc2 = obtener_descriptor(img2)
    
    print("Se muestran los matching por el criterio de Fuerza Bruta + CrossCheck: ")
    matchs_brute_force(img1, kpts1, desc1, img2, kpts2, desc2)
    
    print()
    input("Pulse una tecla para continuar...")
    print()
    
    print("Se muestran los matching por el criterio de Lowe-Average-2NN: ")
    matchs_lowe_average2nn(img1, kpts1, desc1, img2, kpts2, desc2)



    
"""

    FUNCIONES EJERCICIO 3

""" 
# Función que recorta la parte sobrante de un canvas.
# Doc: https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

def recorte_panorama(canvas):

    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # El tercer parametro se ha establecido a 0 para que solo coja el contorno de la imagen
    # y no muchos contornitos pequeños dentro del propio mosaico.
    ret, thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,1, 2)
    
    for c in contours:
        # Sea (x,y) la coordenada de arriba a la izquierda del rectangulo
        # y (w,h) su ancho y alto.
         x,y,w,h = cv2.boundingRect(c)
         if w>5 and h>10:
             cv2.rectangle(canvas,(x,y),(x+w,y+h),(155,155,0),2)
             
    recorte = canvas[y:y+h, x:x+w]
             
    mostrarimagen(recorte, "Mosaico de N imagenes")
    


def construccion_panorama(lista_img, canvas_width, canvas_height):
    
    # Elegimos la imagen central
    centro = len(lista_img)//2
    central = lista_img[centro]
    
    # Creamos un canvas (canvas_width x canvas_height) de color negro
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    # Creamos la homografia inicial (H0=, la cual es una homografia de traslacion al canvas
    # De esta manera tendremos la imagen en el centro del canvas
    h0 = np.array([[1,0, canvas_width//2 - central.shape[1]//2],
                   [0,1, canvas_height//2 - central.shape[0]//2],
                   [0,0,1]], dtype=np.float64)
    
    homografia_0 = np.copy(h0)

    # Parte derecha del mosaico
    for i in reversed(range(centro, len(lista_img)-1)):
        dst = lista_img[i] # Esta es la imagen destino
        src = lista_img[i+1] # Esta es la imagen fuente
        
        # Obtenemos los Keypoints y descriptores utilizando SIFT
        kpts1, desc1 = obtener_descriptor(dst)
        kpts2, desc2 = obtener_descriptor(src)
    
        matcher2 = cv2.BFMatcher_create(cv2.NORM_L2)
    
        # Obtenemos los matches
        mchts2 = matcher2.knnMatch(desc1,desc2,k=2)
           
        matchesMask = []
    
        for m1,m2 in mchts2:
            # David Lowe propone que para establecer si hay match o no el criterio de:
            # m es mejor match si y solamente si d(m1) < 0.8*d(m2), siendo d la distancia
            if m1.distance < 0.8*m2.distance:
                matchesMask.append(m1)
                
        kpts_match_dest = []
        kpts_match_src = []
        
        # Obtener coordenadas de los keypoints de los matches  
        # https://docs.opencv.org/3.4/d4/de0/classcv_1_1DMatch.html
        # Doc: https://stackoverflow.com/questions/13318853/opencv-drawmatches-queryidx-and-trainidx
        for m in matchesMask:
            kpts_match_dest.append(kpts1[m.queryIdx].pt)
            kpts_match_src.append(kpts2[m.trainIdx].pt)
        
        kpts_match_dest = np.array(kpts_match_dest, dtype=np.float32)
        kpts_match_src = np.array(kpts_match_src, dtype=np.float32)
        
        
        # Calculamos la homografía entre cada par de imágenes.
        # El método devuelve la homografía y una máscara, la cual ignoraremos.
        # Doc: https://shimat.github.io/opencvsharp_docs/html/38333149-7bd7-5ddd-eadf-3401d0efab12.htm
        homografia, _ = cv2.findHomography(kpts_match_src, kpts_match_dest, cv2.RANSAC, 3)
        
        # multiplicamos las homografias para qe se aplien las transformaciones correctamente en el resultado
        copia_homografia = np.dot(homografia_0, homografia)
        
        # Añadimos la imagen al resultado con warpPerspective
        canvas = cv2.warpPerspective(src, copia_homografia, (canvas_width, canvas_height), canvas, borderMode = cv2.BORDER_TRANSPARENT)
    

    print()

    # Parte izquerda del mosaico
    for i in reversed(range(1, centro+1)):
        dst = lista_img[i] # Esta es la imagen destino
        src = lista_img[i-1] # Esta es la imagen fuente    
        
        # Obtenemos los Keypoints y descriptores utilizando SIFT
        kpts1, desc1 = obtener_descriptor(dst)
        kpts2, desc2 = obtener_descriptor(src)
    
        matcher2 = cv2.BFMatcher_create(cv2.NORM_L2)
    
        # Obtenemos los matches
        mchts2 = matcher2.knnMatch(desc1,desc2,k=2)
           
        matchesMask = []
    
        for m1,m2 in mchts2:
            # David Lowe propone que para establecer si hay match o no el criterio de:
            # m es mejor match si y solamente si d(m1) < 0.8*d(m2), siendo d la distancia
            if m1.distance < 0.8*m2.distance:
                matchesMask.append(m1)
                
        kpts_match_dest = []
        kpts_match_src = []
        
        # Obtener coordenadas de los keypoints de los matches        
        for m in matchesMask:
            kpts_match_dest.append(kpts1[m.queryIdx].pt)
            kpts_match_src.append(kpts2[m.trainIdx].pt)
        
        kpts_match_dest = np.array(kpts_match_dest, dtype=np.float32)
        kpts_match_src = np.array(kpts_match_src, dtype=np.float32)
              
        homografia, _ = cv2.findHomography(kpts_match_src, kpts_match_dest, cv2.RANSAC, 5)
        # multiplicamos las homografias para qe se aplien las transformaciones correctamente en el resultado
        copia_homografia = np.dot(homografia_0, homografia)
        
        # Añadimos la imagen al resultado con warpPerspective
        canvas = cv2.warpPerspective(src, copia_homografia, (canvas_width, canvas_height), dst=canvas, borderMode = cv2.BORDER_TRANSPARENT)

    
    # Poner imagen central en el mosaico
    canvas= cv2.warpPerspective(central,h0,(canvas_width, canvas_height),dst=canvas,borderMode = cv2.BORDER_TRANSPARENT)  
    mostrarimagen(canvas, "Mosaico 3 imagenes")


"""

    FUNCIONES BONUS 1

""" 

# Como en el ejercicio 1 guardamos los delta de cada extremo
# podemos calcular facilmente la octava con este valor
# Funcion que calcula una lista de los valores de delta por cada octava

def calcula_octava():
    lista_deltas = []
    n_octavas = 4
    delta = 0.5 # Valor del delta inicial
    
    
    for i in range(n_octavas):
        lista_deltas.append(delta)
        delta = delta * 2
        
    return lista_deltas

# Funcion que obtiene el valor del punto w(o,s,m,n)

def obtener_punto(DoG_octavas, octava, loc_interp):

    # La escala del punto será la de la octava o, escala s
    if loc_interp[0] >= 4: loc_interp[0] = 3
    if loc_interp[0] <= 0: loc_interp[0] = 1
       
    escala_elegida = DoG_octavas[int(octava)][int(loc_interp[0])]
    
    # en realidad yo guardo el valor del punto asi que podria ahorrarme
    # todo esto pero lo hago por seguir el algoritmo del paper, que queda mejor
    punto = escala_elegida[int(loc_interp[1])][int(loc_interp[2])] 
    
    return punto

# Funcion que calcula la matriz Hessiana a partir de una octava, una escala
# y las coordenadas de un punto, obteniendo el valor a partir del espacio de escalas de la DoG

def matriz_hessiana(DoG_octavas, octava, loc_interp, mi_punto):
    # punto = w(s, m, n) -> w(escala, x, y)
    s = loc_interp[0]
    x = loc_interp[1]
    y = loc_interp[2]
    
    # obtener_punto = Funcion que obtiene el valor del punto w(o,s,m,n)
    p1_h11 = obtener_punto(DoG_octavas, octava, [s+1,x,y])
    p2_h11 = obtener_punto(DoG_octavas, octava, [s-1,x,y])
    
    p1_h12 = obtener_punto(DoG_octavas, octava, [s+1,x+1,y])
    p2_h12 = obtener_punto(DoG_octavas, octava, [s+1,x-1,y])
    p3_h12 = obtener_punto(DoG_octavas, octava, [s-1,x+1,y])
    p4_h12 = obtener_punto(DoG_octavas, octava, [s-1,x-1,y])
    
    p1_h13 = obtener_punto(DoG_octavas, octava, [s+1,x,y+1])
    p2_h13 = obtener_punto(DoG_octavas, octava, [s+1,x,y-1])
    p3_h13 = obtener_punto(DoG_octavas, octava, [s-1,x,y+1])
    p4_h13 = obtener_punto(DoG_octavas, octava, [s-1,x,y-1])
    
    p1_h22 = obtener_punto(DoG_octavas, octava, [s,x+1,y])
    p2_h22 = obtener_punto(DoG_octavas, octava, [s,x-1,y])
    
    p1_h23 = obtener_punto(DoG_octavas, octava, [s,x+1,y+1])
    p2_h23 = obtener_punto(DoG_octavas, octava, [s,x+1,y-1])
    p3_h23 = obtener_punto(DoG_octavas, octava, [s,x-1,y+1])
    p4_h23 = obtener_punto(DoG_octavas, octava, [s,x-1,y-1])
    
    p1_h33 = obtener_punto(DoG_octavas, octava, [s,x,y+1])
    p2_h33 = obtener_punto(DoG_octavas, octava, [s,x,y-1])
    
    h11 = p1_h11 + p2_h11 - (2*mi_punto)
    h22 = p1_h22 + p2_h22 - (2*mi_punto)
    h33 = p1_h33 + p2_h33 - (2*mi_punto)
    h12 = (p1_h12 - p2_h12 - p3_h12 + p4_h12)/4
    h13 = (p1_h13 - p2_h13 - p3_h13 + p4_h13)/4    
    h23 = (p1_h23 - p2_h23 - p3_h23 + p4_h23)/4
    
    
    hessiana = [[h11,h12,h13],
                [h12,h22,h23],
                [h13,h23,h33]]
    
    return hessiana

# Funcion que calcula el kernel jacobiano (gradiente 3D)

def jacobiano(DoG_octavas, octava, loc_interp, mi_punto):
    # punto = w(s, m, n) -> w(escala, x, y)
    s = loc_interp[0]
    x = loc_interp[1]
    y = loc_interp[2]
    
    
    h11_1 = obtener_punto(DoG_octavas, octava, [s+1,x,y])
    h11_2 = obtener_punto(DoG_octavas, octava, [s-1,x,y])
    
    h12_1 = obtener_punto(DoG_octavas, octava, [s,x+1,y])
    h12_2 = obtener_punto(DoG_octavas, octava, [s,x-1,y])
    
    h13_1 = obtener_punto(DoG_octavas, octava, [s,x,y+1])
    h13_2 = obtener_punto(DoG_octavas, octava, [s,x,y-1])
    
    h11 = (h11_1 - h11_2)/2
    h12 = (h12_1 - h12_2)/2
    h13 = (h13_1 - h13_2)/2
    
    jacobiana = [h11,h12,h13]
    
    return jacobiana

# Funcion que calcula la interpolacion cuadratica de un punto
# Basada en el algoritmo 7 del paper The Anatomy of SIFT

def interpolacion_cuadratica(DoG_octavas, octava, loc_interp):
    
    mi_punto = obtener_punto(DoG_octavas, octava, loc_interp)

    
    hessiana = matriz_hessiana(DoG_octavas, octava, loc_interp, mi_punto)
    # calculo inversa de la hessiana
    hessiana_inv = np.linalg.inv(hessiana)
   
    jacobiana = jacobiano(DoG_octavas, octava, loc_interp, mi_punto)
    # calculo traspuesta de la jacobiana
    jacobiana_tras = np.transpose(jacobiana)
    
     
    # Desplazamiento desde el centro del extremo 3D interpolado.
    desp = -(np.dot(hessiana_inv,jacobiana))   
    
    operacion1 = (1/2)*jacobiana_tras
    operacion2 = np.dot(operacion1,hessiana_inv)
    operacion3 = np.dot(operacion2, jacobiana)
    
    #Valor del extremo 3D interpolado
    punto_interpolado = mi_punto - operacion3
    
    
    return desp, punto_interpolado


# Funcion que calcula el refinamiento de los extremos encontrado en el ejercicio 1
# Basada en el algoritmo 6 del paper The Anatomy of SIFT

def interpolacion_keypoints(imagen, DoG_octavas, extremos):
    keypoints_interpolados = []
    
    lista_deltas = calcula_octava() # Calculamos octava a partir de delta
    
    for i in range(len(extremos)):

        # mi extremo tiene la forma (valor,x,y,sigma_k,delta,escala)
        # La octava es la posicion de la lista donde se encuentre delta
        # Doc: https://numpy.org/doc/stable/reference/generated/numpy.where.html
        delta = extremos[i][4]
        o = np.where(lista_deltas == delta )[0]
        s = extremos[i][5] 
        x = extremos[i][1]
        y = extremos[i][2]
        
        sigma_min = 0.8
        delta_min = 0.5

        
        # Inicializar la localizacion de interpolacion
        loc_interp = [s, x, y]

        desp = [np.inf, np.inf, np.inf] # Lo inicializamos a INF porque queremos que sea < que 0.6
        maximo = max(desp)
        
        contador = 0
        
        # mientras que no sea menor que 0.6 o durante 5 intentos
        while maximo > 0.6 and contador < 5:

            # Calcular la ubicacion extrema y el valor de la funcion cuadratica local
            desp, extremo_interpolado = interpolacion_cuadratica(DoG_octavas, o, loc_interp)
            maximo = max(desp)

            # Calculamos las correspondientes coordenadas absolutas
            # donde desp = [alpha1, alpha2, alpha3]
            sigma = round((delta/delta_min)*sigma_min*2**((desp[0]+s)/5))
            x_abs = round(delta*(desp[1]+x))
            y_abs = round(delta*(desp[2]+y))
        
            
            # Actualizamos la posicion interpolada
            s = s + desp[0]
            x = x + desp[1]
            y = y + desp[2]
            
            loc_interp = [s, x, y]
            
            maximo = max(desp)
            contador = contador +1

                       
        if maximo < 0.6:
            extremo_interpolado = [int(o),s,x,y,sigma,x_abs,y_abs,extremo_interpolado,delta]
            keypoints_interpolados.append(extremo_interpolado)
    
    
    print("Se muestran los Keypoints refinados")
    
    keypoints = []
    # Convertimos los extremos en Keypoints con cv2.KeyPoint
    for i in range(len(keypoints_interpolados)):
        #keypoints_interpolados[i][2] = keypoints_interpolados[i][2] * keypoints_interpolados[i][8]
        #keypoints_interpolados[i][3] = keypoints_interpolados[i][3] * keypoints_interpolados[i][8]
        
        #keypoints.append(cv2.KeyPoint(keypoints_interpolados[i][3], keypoints_interpolados[i][2], size =((keypoints_interpolados[i][4]*6)*2)))
        keypoints.append(cv2.KeyPoint(keypoints_interpolados[i][6], keypoints_interpolados[i][5], size =((keypoints_interpolados[i][4]*6)*2)))
        
    
    imagen_resultado = cv2.drawKeypoints(imagen, keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    mostrarimagen(imagen_resultado, "Refinamiento de keypoints")
    
   

    
    
"""

    FUNCIONES BONUS 2

"""     
    
def construccion_panorama_grande(lista_img, canvas_width, canvas_height, centro):
    
    # Elegimos la imagen central
    central = lista_img[centro]
    
    # Creamos un canvas (canvas_width x canvas_height) de color negro
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    # Creamos la homografia inicial (H0=, la cual es una homografia de traslacion al canvas
    # De esta manera tendremos la imagen en el centro del canvas
    h0 = np.array([[1,0, canvas_width//2 - central.shape[1]//2],
                   [0,1, canvas_height//2 - central.shape[0]//2],
                   [0,0,1]], dtype=np.float64)
    
    homografia_0 = np.copy(h0)
    
    homografias_derecha = []
    homografias_izquierda = []
    

    # Se hace el calculo de las homografias de la parte DERECHA del mosaico
    for i in range(centro, len(lista_img)-1):
        dst = lista_img[i] # Esta es la imagen destino
        src = lista_img[i+1] # Esta es la imagen fuente
        
        # Obtenemos los Keypoints y descriptores utilizando SIFT
        kpts1, desc1 = obtener_descriptor(dst)
        kpts2, desc2 = obtener_descriptor(src)
    
        matcher2 = cv2.BFMatcher_create(cv2.NORM_L2)
    
        # Obtenemos los matches
        mchts2 = matcher2.knnMatch(desc1,desc2,k=2)
           
        matchesMask = []
    
        for m1,m2 in mchts2:
            # David Lowe propone que para establecer si hay match o no el criterio de:
            # m es mejor match si y solamente si d(m1) < 0.8*d(m2), siendo d la distancia
            if m1.distance < 0.8*m2.distance:
                matchesMask.append(m1)
                
        kpts_match_dest = []
        kpts_match_src = []
        
        # Obtener coordenadas de los keypoints de los matches        
        for m in matchesMask:
            kpts_match_dest.append(kpts1[m.queryIdx].pt)
            kpts_match_src.append(kpts2[m.trainIdx].pt)
        
        kpts_match_dest = np.array(kpts_match_dest, dtype=np.float32)
        kpts_match_src = np.array(kpts_match_src, dtype=np.float32)
              
        homografia, _ = cv2.findHomography(kpts_match_src, kpts_match_dest, cv2.RANSAC, 3)
        
        # Una vez que ya haya una lista calculada e incluida en la lista
        # La homografía debe mutiplicarse con todas las homografías hasta llegar a la foto central.
        # Se multiplica con la ultima posición de la lista ya que esta va conteniendo las multiplicaciones de las homografias.
        if len(homografias_derecha) > 0:
            homografia = np.dot(homografias_derecha[len(homografias_derecha)-1],homografia)
        
        # Añadimos la homografia a la lista              
        homografias_derecha.append(homografia)
        
        copia_homografia = np.dot(homografia_0, homografia)
        
        # Añadimos la imagen al resultado con warpPerspective
        canvas = cv2.warpPerspective(src, copia_homografia, (canvas_width, canvas_height), canvas, borderMode = cv2.BORDER_TRANSPARENT)
        

        
    # Se hace el calculo de las homografias de la parte IZQUIERDA del mosaico
    for i in reversed(range(1, centro+1)):
        dst = lista_img[i] # Esta es la imagen destino
        src = lista_img[i-1] # Esta es la imagen fuente    
        
        # Obtenemos los Keypoints y descriptores utilizando SIFT
        kpts1, desc1 = obtener_descriptor(dst)
        kpts2, desc2 = obtener_descriptor(src)
    
        matcher2 = cv2.BFMatcher_create(cv2.NORM_L2)
    
        # Obtenemos los matches
        mchts2 = matcher2.knnMatch(desc1,desc2,k=2)
           
        matchesMask = []
    
        for m1,m2 in mchts2:
            # David Lowe propone que para establecer si hay match o no el criterio de:
            # m es mejor match si y solamente si d(m1) < 0.8*d(m2), siendo d la distancia
            if m1.distance < 0.8*m2.distance:
                matchesMask.append(m1)
                
        kpts_match_dest = []
        kpts_match_src = []
        
        # Obtener coordenadas de los keypoints de los matches        
        for m in matchesMask:
            kpts_match_dest.append(kpts1[m.queryIdx].pt)
            kpts_match_src.append(kpts2[m.trainIdx].pt)
        
        kpts_match_dest = np.array(kpts_match_dest, dtype=np.float32)
        kpts_match_src = np.array(kpts_match_src, dtype=np.float32)
              
        homografia, _ = cv2.findHomography(kpts_match_src, kpts_match_dest, cv2.RANSAC, 3)
        
        # Una vez que ya haya una lista calculada e incluida en la lista
        # La homografía debe mutiplicarse con todas las homografías hasta llegar a la foto central.
        # Se multiplica con la ultima posición de la lista ya que esta va conteniendo las multiplicaciones de las homografias.
        
        if len(homografias_izquierda) > 0:
            homografia = np.dot(homografias_izquierda[len(homografias_izquierda)-1], homografia)
                      
        homografias_izquierda.append(homografia)
    
    # En el caso de la parte izquierda del mosaico, no se puede hacer todo directamente en un único bucle for
    # ya que las homografías se calculan desde la primera imagen hasta el centro, mientras que las montamos en el resultado
    # de manera inversa al cálculo. Si se hiciera todo en un mismo bucle for no encajarían del todo.
    n=0
        
    for i in reversed(range(len(homografias_izquierda))):
        copia_homografia = np.dot(homografia_0, homografias_izquierda[i]) 
        
        # Añadimos la imagen al resultado con warpPerspective 
        canvas = cv2.warpPerspective(lista_img[n], copia_homografia, (canvas_width, canvas_height), canvas, borderMode = cv2.BORDER_TRANSPARENT)
        n = n+1
    
    # Poner imagen central en el mosaico
    canvas= cv2.warpPerspective(central,h0,(canvas_width, canvas_height),dst=canvas,borderMode = cv2.BORDER_TRANSPARENT)  

    
    return canvas




    



"""

    MAIN DEL PROGRAMA
 """   

filename = './imagenes/Yosemite1.jpg'
yosemite1 = leeimagen(filename, cv2.IMREAD_GRAYSCALE)

filename = './imagenes/Yosemite2.jpg'
yosemite2 = leeimagen(filename, cv2.IMREAD_GRAYSCALE)
  
print("----------- EJERCICIO 1 -----------")
print("IMAGEN: YOSEMITE 1")

sigma_ad = 0.8 # Sigma de adquisición proporcionado en el enunciado del problema

DoG_octavas_y1, top_100_y1 = ejercicio_1(yosemite1, sigma_ad)

input("Pulse una tecla para continuar...")
print()
print("IMAGEN: YOSEMITE 2")

DoG_octavas_y2, top_100_y2 = ejercicio_1(yosemite2, sigma_ad)


input("Pulse una tecla para continuar...")
print()
   
print("----------- EJERCICIO 2 -----------")

random.seed(4) # Establecemos una semilla para que siempre sean las mismas muestras aleatorias

correspondencias(yosemite1, yosemite2)
    
input("Pulse una tecla para continuar...")
print() 


print("----------- EJERCICIO 3 -----------")
print()

# Cargamos las imagenes de Graná ciudad de la Alhambra

filename = './imagenes/IMG_20211030_110413_S.jpg'
grana1 = leeimagen(filename, cv2.IMREAD_COLOR)
grana1_dim = grana1.shape


filename = './imagenes/IMG_20211030_110415_S.jpg'
grana2 = leeimagen(filename, cv2.IMREAD_COLOR)
grana2_dim = grana2.shape

filename = './imagenes/IMG_20211030_110417_S.jpg'
grana3 = leeimagen(filename, cv2.IMREAD_COLOR)
grana3_dim = grana3.shape

# Calculamos el tamaño del canvas
maxi = [grana1_dim[0], grana2_dim[0], grana3_dim[0]]

# Utilizamos una altura algo mayor de la altura de la imagen
canvas_height = max(maxi) + 30

# Como máximo, jamas superara un ancho que sea la suma de las imagenes.
# Sin embargo, como se solapa gran parte de las imagenes, se considera que no superara
# la suma de los anchos de la mitad de las imagenes de la listA.
canvas_width = (grana1_dim[1] + grana2_dim[1])


lista_img = [grana1, grana2, grana3]

print("Mostramos mosaico con tres imagenes de la Alhambra.")
construccion_panorama(lista_img, canvas_width, canvas_height)


input("Pulse una tecla para continuar...")
print()

filename = './imagenes/IMG_20211030_110421_S.jpg'
grana1 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110425_S.jpg'
grana2 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110426_S.jpg'
grana3 = leeimagen(filename, cv2.IMREAD_COLOR)

lista_img = [grana1, grana2, grana3]

print("Mostramos mosaico con tres imagenes de la ciudad de Granada.")
construccion_panorama(lista_img, canvas_width, canvas_height)



input("Pulse una tecla para continuar...")
print()

print("----------- BONUS B1 -----------")
print()


print("Se calcula el refinamiento de los maximos obtenidos en el ejercicio 1")
print("IMAGEN: YOSEMITE 1")
print()
interpolacion_keypoints(yosemite1, DoG_octavas_y1, top_100_y1)

input("Pulse una tecla para continuar...")
print()

print("IMAGEN: YOSEMITE 2")
print()

interpolacion_keypoints(yosemite2, DoG_octavas_y2, top_100_y2)


input("Pulse una tecla para continuar...")
print()
print("----------- BONUS B2 - Opcion A -----------")
print()

filename = './imagenes/IMG_20211030_110410_S.jpg'
grana0 = leeimagen(filename, cv2.IMREAD_COLOR)
grana0_dim = grana0.shape

filename = './imagenes/IMG_20211030_110413_S.jpg'
grana1 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110415_S.jpg'
grana2 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110417_S.jpg'
grana3 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110418_S.jpg'
grana4 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110420_S.jpg'
grana5 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110421_S.jpg'
grana6 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110425_S.jpg'
grana7 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110426_S.jpg'
grana8 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110428_S.jpg'
grana9 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110431_S.jpg'
grana10 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110433_S.jpg'
grana11 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110434_S.jpg'
grana12 = leeimagen(filename, cv2.IMREAD_COLOR)

filename = './imagenes/IMG_20211030_110436_S.jpg'
grana13 = leeimagen(filename, cv2.IMREAD_COLOR)


lista_img = [grana0, grana1, grana2, grana3, grana4, grana5, grana6, grana7, grana8, grana9, grana10, grana11, grana12, grana13]


canvas_width = grana0_dim[0] * 16
canvas_height= grana0_dim[1] * 3

centro = len(lista_img)//2
print("¡¡ Los canvas han sido recortados!!")
print()

print("Construcción mosaico de N imagenes con el centro en la imagen 7.")
canvas = construccion_panorama_grande(lista_img, canvas_width, canvas_height, centro)
recorte_panorama(canvas)


input("Pulse una tecla para continuar...")
print()

centro = (len(lista_img)//2) - 1
print("Construcción mosaico de N imagenes con el centro en la imagen 6.")
canvas = construccion_panorama_grande(lista_img, canvas_width, canvas_height, centro)
recorte_panorama(canvas)

input("Pulse una tecla para continuar...")
print()

centro = (len(lista_img)//2) - 2
print("Construcción mosaico de N imagenes con el centro en la imagen 5.")
canvas = construccion_panorama_grande(lista_img, canvas_width, canvas_height, centro)
recorte_panorama(canvas)


