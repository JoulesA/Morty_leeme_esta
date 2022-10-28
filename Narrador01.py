import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from PIL import Image

# FUNCIONES para la red neuronal 
def hardlim(n):
    if n > 0 :
        out = 1
    else: 
        out = 0

    return out

# FUNCIONES para procesamiento de imagenes
def rgb2g(im):
    n_img = 0.299 * im[:,:,0] + 0.587 * im[:,:,1] + 0.114 * im[:,:,2]
    return n_img

def binarize(im, treshold = 200):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] <= treshold:
                im[i][j] = 0
            else:
                im[i][j] = 255
    return im

# Carga Directamente en gris
def Load_Gray(name):
    img = Image.open(name)
    img_arr = np.asarray(img)
    print(type(img_arr))
    print(img_arr.shape)
    img_gray = rgb2g(img_arr)
    print(img_gray.shape)
    
    imbin = binarize(img_gray)  # Imagen binarizada
    return imbin

def Get_Letters_vector(imagen):
    imbin=Load_Gray(imagen)
    
    filsum = []
    for i in range(imbin.shape[0]):
        filsum.append(sum(imbin[i,:]))
    
    # Crop row
    row_tresh = sum(imbin[0,:])
    crop_vec = []
    for i in range(1,len(filsum)):
        if filsum[i-1]== row_tresh and filsum[i]<row_tresh:
            crop_vec.append(i)
        elif filsum[i]== row_tresh and filsum[i-1]<row_tresh: 
            crop_vec.append(i)
    
    # Quita espacios entre acentos y la cosa de la ñ
    counter_pop = 0
    for i in range(1,len(crop_vec)):
        if (crop_vec[i-counter_pop]-crop_vec[i-1-counter_pop])<=4:
            crop_vec.pop(i-1-counter_pop)
            crop_vec.pop(i-1-counter_pop)
            counter_pop += 2
            
    
    # Row cropted
    rows = []
    for i in range(0,len(crop_vec),2):
        rows.append(imbin[crop_vec[i]:crop_vec[i+1],:])
        
    
    # Corte letras
    # Para cada renglon
    letras = []
    for k in range(len(rows)):
        
        colsum = []
        col_tresh = sum(rows[k][:,0])
        
        
        for i in range(rows[k].shape[1]):
            colsum.append(sum(rows[k][:,i]))
        
        # Donde corta
        crop_vec = []
        for i in range(1,len(colsum)):
            if colsum[i-1] == col_tresh and colsum[i]<col_tresh:
                crop_vec.append(i)
            elif colsum[i] == col_tresh and colsum[i-1]<col_tresh:
                crop_vec.append(i)
        
        # Hora de ver espacios y quitar los espacios en blanco entre letras
        for i in range(0,len(crop_vec),2):
            letras.append(rows[k][:,crop_vec[i]:crop_vec[i+1]])
            
    # De imagen a vector
    letras_vec = []
    for i in range(len(letras)):
        vec = np.reshape(letras[i],((letras[i].shape[0]*letras[i].shape[1]),1))
        letras_vec.append(vec)
    
    return letras_vec

Get_Letters_vector('Letras.bmp')