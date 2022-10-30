import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from PIL import Image
#from gtts import gTTS
import webbrowser 

# FUNCIONES para la red neuronal 
def Hardlim(n):
    out=[]
    for i in n:
        if i >= 0 :
            out.append(1)
        else: 
            out.append(0)
    return out

def Relu(n):
    out=[]
    for i in n:
        if i >= 0 :
            out.append(i)
        else: 
            out.append(0)
    return out

def Compet(n):
    out = np.where(n == max(n),1 ,0)
    return out 

def Diccionario(v):
    if v[0] == 1:
        print('A')
    elif v[1] == 1:
        print('B')
    elif v[2] == 1:
        print('C')
    elif v[3] == 1:
        print('D')
    elif v[4] == 1:
        print('E')
    elif v[5] == 1:
        print('F')
    elif v[6] == 1:
        print('G')
    elif v[7] == 1:
        print('H')
    elif v[8] == 1:
        print('I')
    elif v[9] == 1:
        print('J')
    elif v[10] == 1:
        print('K')
    elif v[11] == 1:
        print('L')
    elif v[12] == 1:
        print('M')
    elif v[13] == 1:
        print('N')
    elif v[14] == 1:
        print('O')
    elif v[15] == 1:
        print('P')
    elif v[16] == 1:
        print('Q')
    elif v[17] == 1:
        print('R')
    elif v[18] == 1:
        print('S')
    elif v[19] == 1:
        print('T')
    elif v[20] == 1:
        print('U')
    elif v[21] == 1:
        print('V')
    elif v[22] == 1:
        print('W')
    elif v[23] == 1:
        print('X')
    elif v[24] == 1:
        print('Y')
    elif v[25] == 1:
        print('Z')
    elif v[26] == 1:
        print('a')
    elif v[27] == 1:
        print('b')
    elif v[28] == 1:
        print('c')
    elif v[29] == 1:
        print('d')
    elif v[30] == 1:
        print('e')
    elif v[31] == 1:
        print('f')
    elif v[32] == 1:
        print('g')
    elif v[33] == 1:
        print('h')
    elif v[34] == 1:
        print('i')
    elif v[35] == 1:
        print('j')
    elif v[36] == 1:
        print('k')
    elif v[37] == 1:
        print('l')
    elif v[38] == 1:
        print('m')
    elif v[39] == 1:
        print('n')
    elif v[40] == 1:
        print('ñ')
    elif v[41] == 1:
        print('o')  
    elif v[42] == 1:
        print('p')
    elif v[43] == 1:
        print('q')
    elif v[44] == 1:
        print('r')
    elif v[45] == 1:
        print('s')
    elif v[46] == 1:
        print('t')
    elif v[47] == 1:
        print('u')
    elif v[48] == 1:
        print('v')
    elif v[49] == 1:
        print('w')
    elif v[50] == 1:
        print('x')
    elif v[51] == 1:
        print('y')
    elif v[52] == 1:
        print('z')
    elif v[53] == 1:
        print('á')
    elif v[54] == 1:
        print('é')
    elif v[55] == 1:
        print('í')
    elif v[56] == 1:
        print('ó')
    elif v[57] == 1:
        print('ú')
        
        
    
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
    
    return letras_vec,letras

class INSTAR:
    def __init__(self):
        self.w=[]
        self.bias=[]
        self.pat=[]
    
    def Load_W(self,vectors, similitude = 0.90):
        lv_size =[]
        for i in range(len(vectors)):
            lv_size.append(vectors[i].shape[0]) 
        w_max_size = max(lv_size)
        
        # W [Letras,NCaract] and bias 
        self.bias=[]
        # Parecido es un factor entre 0 y uno para ver que tanto se parecen 
        self.w = np.zeros((len(vectors),w_max_size))
        for i in range(len(vectors)):
            norma = 0
            for k in range(vectors[i].shape[0]):
                self.w[i,k] = vectors[i][k,0]
                norma += (vectors[i][k,0])*(vectors[i][k,0])
            self.bias.append(norma*similitude)
        self.bias = np.array(self.bias)
        self.bias = np.reshape(self.bias,(len(self.bias),1))
    
    def Test(self,pattern):

        pat = np.zeros((self.w.shape[1],1))
        for i in range(len(pattern)):
            if i <= self.w.shape[1]-1:
                pat[i,0] = pattern[i]
        pattern = pat
                
        out = self.w @ pattern - self.bias
        # out = Hardlim(out)
        # out = Relu(out) # Se obtiene mas o menos lo mismo 
        out = Compet(out)        # Compet
        
        return out

# Obtain the vector of the letters
# letters = Get_Letters_vector('Letras.bmp')
letters,lshapes = Get_Letters_vector('ABC.jpg')

# Create the instar
model = INSTAR()
# Load the W whit the vectors of letters
model.Load_W(letters)
# Test W
outs = []
for i in range(len(letters)):
    sal = model.Test(model.w[i].T)
    outs.append(sal)
    # print(sal)
outs = np.array(outs)
    
texto,tshapes = Get_Letters_vector('LETRAS.jpg')
outst = []
for i in range(len(texto)):
    sal = model.Test(texto[i])
    outst.append(sal)
    Diccionario(sal)
outst = np.array(outst)
    
#print (outs)