# Morty_leeme_esta
Programa "Text to speech" usando redes neuronales en Python 

el programa `narrador01.py` usa funciones y unaclase para la creacion de los pesos W, ademas cuenta con dos funciones, la primera recibe la matriz con los vectores generados por las letras, mientras que el segundo directamente hace la convolución ($w @ pat - bias$).

el programa `narrador.py` tiene lo mismo que `narrador.py` solo que ese no esta con funciones, esta como te gusta **guiño guiño** solo que no tiene la parte de la instar completa

Creo que hay un detalle en el reconocedor ya que encuentra varias coincidencias similares asi que seria bueno añadir una seccion de competencia para que a la salida solo una salida sea activada.

## Procesamiento de imagenes
la funcion de obtener vectores a partir de imagenes lo que hace, es exactamente eso, segmenta la imagen en renglones y luego en letras.
Primero pasa la imagen de colores a escala de grises y despues binariza, 0 o 255, despues hace la suma en filas y guarda eso en un vector, despues revisamos el vector para encontrar donde se dan los cambios y solo consideramos cuando pasa del maximo a otro valor o de un valor distinto del maximo al maximo, es decir cuando pasa de fila blanca a una con algun dato en 0 o de alguna fila con un 0 a 255, eliminamos los espacios menores a un minimo para aliminar los espacios que puede haber entre la cosa que tiene la ñ y la forma n o entre el acento y la letra, despues se hace un corte de renglones y se aplica un proceso similar pero en columnas para obtener las letras, despues se ocupa un reshape para hacer de las imagenenes de letra un vector y eso es lo que se entrega.

## De la red neuronal instar
Para cuando se inicializa w se hace con una matriz de ceros del mismo tamaño que la cantidad de letras y el tamaño maximo de vector que entro, aqui puede que se pueda mejorar si en vez de hacer matriz de ceros se hace de 255 para variar la info.
Seria conveniente añadir una compet al final, pero primero haz el cambio de la matriz de ceros para ver si mejora y no trabajes de mas.
