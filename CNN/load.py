import os
import cv2
from PIL import Image
import numpy as np
 
data=[]
labels=[]

# alugbati 0
alugbati = os.listdir(os.getcwd() + "/CNN/data/Alugbati/")
for x in alugbati:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Alugbati/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

# basil 1
basil = os.listdir(os.getcwd() + "/CNN/data/Basil/")
for x in basil:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Basil/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

# blueternate 2
blueternate = os.listdir(os.getcwd() + "/CNN/data/Blueternate/")
for x in blueternate:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Blueternate/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)

# Chess nut 3
chessNut = os.listdir(os.getcwd() + "/CNN/data/Chess nut/")
for x in chessNut:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Chess nut/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(3)

# Chilli pepper 4
chilliPepper = os.listdir(os.getcwd() + "/CNN/data/Chilli pepper/")
for x in chilliPepper:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Chilli pepper/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(4)

# Chives 5
chives = os.listdir(os.getcwd() + "/CNN/data/Chives/")
for x in chives:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Chives/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(5)

# Guava 6
guava = os.listdir(os.getcwd() + "/CNN/data/Guava/")
for x in guava:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Guava/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(6)

# Guyabano 7
guyabano = os.listdir(os.getcwd() + "/CNN/data/Guyabano/")
for x in guyabano:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Guyabano/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(7)

# Lagundi 8
lagundi = os.listdir(os.getcwd() + "/CNN/data/Lagundi/")
for x in lagundi:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Lagundi/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(8)

# Lemon 9
lemon = os.listdir(os.getcwd() + "/CNN/data/Lemon/")
for x in lemon:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Lemon/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(9)

# Mexican Mint 10
mexicanMint = os.listdir(os.getcwd() + "/CNN/data/Mexican Mint/")
for x in mexicanMint:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Mexican Mint/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(10)

# Miracle Fruit 11
miracleFruit = os.listdir(os.getcwd() + "/CNN/data/Miracle Fruit/")
for x in miracleFruit:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Miracle Fruit/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(11)
    
# Moringa Oleifera 12
moringaOleifera = os.listdir(os.getcwd() + "/CNN/data/Moringa oleifera/")
for x in moringaOleifera:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Moringa oleifera/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(12)

# Passion fruit 13
passionFruit = os.listdir(os.getcwd() + "/CNN/data/Passion fruit/")
for x in passionFruit:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Passion fruit/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(13)

# Plant 14
plant = os.listdir(os.getcwd() + "/CNN/data/Plant/")
for x in plant:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Plant/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(14)

# Rose mary 15
roseMary = os.listdir(os.getcwd() + "/CNN/data/Rose mary/")
for x in roseMary:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Rose mary/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(15)

# Stevia 16
stevia = os.listdir(os.getcwd() + "/CNN/data/Stevia/")
for x in stevia:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Stevia/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(16)

# Stevia 17
taragon = os.listdir(os.getcwd() + "/CNN/data/taragon/")
for x in taragon:
    imag=cv2.imread(os.getcwd() + "/CNN/data/taragon/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(17)

# tawa tawa 18
tawaTawa = os.listdir(os.getcwd() + "/CNN/data/tawa tawa/")
for x in tawaTawa:
    imag=cv2.imread(os.getcwd() + "/CNN/data/tawa tawa/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(18)

# Wachichao 19
wachichao = os.listdir(os.getcwd() + "/CNN/data/Wachichao/")
for x in wachichao:
    imag=cv2.imread(os.getcwd() + "/CNN/data/Wachichao/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(19)

plants=np.array(data)
labels=np.array(labels)
# 
np.save("plants",plants)
np.save("labels",labels)