import cv2
from math import cos, sin, pi
import matplotlib.pyplot as plt #carga la librería para graficar
import numpy as np

#Funcion de la rotacion
#En esta funcion usamos como se muestra en el documento 
#la formula de rotaciones en trasformaciones lineales
def rotacion(a):
    x1 = a[0]
    x2 = a[1]
    a[0] = cos(0.08726646)*(x1)+(-sin(0.08726646))*x2
    a[1] = (x1*sin(0.08726646))+(x2*cos(0.08726646))
    return a
 
#utilizamos la libreria opencv-python para lo cual es necesarion instalarla:
#pip install opencv-python
#Tambien tenemos que instalar la libreria matplotlib:
#pip install matplotlib
#Sobra decir que es necesario tener python y pip instalados en el 
#equipo para correr adecuadamente el programa, tambien tener la imagen
#descargada y en la misma carpeta o en su defecto cambiar la ruta a continuacion
img = cv2.imread('img_prueba.jpeg')
rows,cols,ch = img.shape
 
#Usamos los puntos de referencia que determinamos en el documento con la ayuda de
#https://www.adobe.com/es/express/feature/image/editor
input_matriz=[[-26.62,39.33],[-23.37,-25.33],[16.42,39.33]]

#Usamos la funcion float32 la cual acepta como parametro una matriz con los
#puntos de referencia iniciales y lo guardamos en pts1, luego hacemos lo mismo
#pero con la rotacion lista con la funcion que diseñamos anteriormente
pts1 = np.float32(input_matriz)
pts2 = np.float32([rotacion(input_matriz[0]),rotacion(input_matriz[1]),input_matriz[2]])
 
#Pasamos como parametros nuestras 2 matrices con los puntos de referencia para graficar
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
 
#Imprimimos la imagen de entrada y la de salida con los calculos aplicados
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

#By Juan Pablo Rivera Alvarez Celula 24