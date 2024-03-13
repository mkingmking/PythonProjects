import cv2
import numpy as np
import sys
import random
import matplotlib.pyplot as plt

kernel = np.ones((3, 3), np.uint8) ## notre kernel pour les transformations erode et dilute

img_black = np.ones((350, 500, 3), dtype = np.uint8)
img_black = 255*img_black



img = cv2.imread("prj.png") ## ouverture de l'image
img_gray = cv2.imread("prj.png", cv2.IMREAD_GRAYSCALE)  ## faire l'image grayscale



  ###### chroma keying
image_copy = img.copy() ## on a besoin de l'image avec couleurs.

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

lower_orange = np.array([130, 80, 10])     ##[R value, G value, B value]
upper_orange = np.array([220, 140, 60])   ##  l'intervalle pour trouver  l'orange dans l'image

#cette fonction fait les pixels blanc ou noir en considerant leur valeurs. Si ils sont plus grand que lower et plus 
#petit que upper, ils sont noir. Sinon, ils sont blanc.
mask = cv2.inRange(image_copy, lower_orange,upper_orange ) 

masked_image = np.copy(image_copy)




### trouver une piece de puzzle: 
img_puzzle = np.zeros((200,120,3), np.uint8)

for i in range(200):
    for j in range(120):
        img_puzzle[i,j] = mask[i +  148,j + 459] 

cv2.imshow("vingenette", img_puzzle)



###image contouring::
img_puzzle2 = cv2.cvtColor(img_puzzle, cv2.COLOR_BGR2GRAY) ##" color bgr-Gray"


ret,thresh1 = cv2.threshold(img_puzzle2,127,255,cv2.THRESH_BINARY)
#cette fonction permet de trouver les points de contour. 
#Dans ce projet, on n'utilise pas hierarchy car on a seulement besoin de mur exterieure.
contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
i = 0

img_puzzle_copy = img_puzzle.copy() ## une copie pour peintre l'approximation
for contour in contours2:
  
   
    
    if i == 0:
        i = 1
        continue
  
    # cv2.approxPloyDP() fonction pour approximer le shape
    epsilon = 0.04 ## differents epsilon valeurs donnent resultats differents .

    # la fonction qui permet l'approximation de contour:
    approx = cv2.approxPolyDP(
    contour, epsilon * cv2.arcLength(contour, True), True)
    print(len(approx))

    ## peintre l'approximation:
    cv2.drawContours(img_puzzle_copy, [approx], 0, (0,255,0), 3)
    
    ## peintre le contour:
    cv2.drawContours(img_puzzle, [contour], 0, (200, 0, 255), 5)
    



cv2.imshow('SIMPLE Approximation contours', img_puzzle)
cv2.imshow('Approximation de shape par contours', img_puzzle_copy)
cv2.waitKey(0)

  
# Utilisant cv2.erode() method:
## au lieu de les deux, on utilise morphologyEx au dessous qui est une fonction qui fait errode et dilate en meme temps.
"""
image_errode = cv2.erode(img_gray, kernel, iterations = 10) 
image_dilate = cv2.dilate(image_errode, kernel, iterations = 10)
"""


#valeur initial pour  creer un trackbar:
valeur = 5
def fonctionTrackbar(v):
    global valeur
    valeur = v

i = 0
ksize = (3,3)
gradient = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)

while True: # on cree un menu avec les buttons q k l et z.
    key = cv2.waitKey(30) & 0x0FF

    gradient = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel, iterations = i)
    #gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel, iterations = 1)
    ### ca marche bien avec i = 4 / 5 aussi.
    gradient = cv2.medianBlur(gradient, 5)    
    #gradient = cv2.GaussianBlur(gradient, ksize,0) 
    cv2.imshow("apres transformation morphologique", gradient)
    cv2.namedWindow('apres transformation morphologique')
    cv2.createTrackbar('Valeur','apres transformation morphologique',valeur,5,fonctionTrackbar)
    
    
    if key == 27 or key==ord('q'):
        print('arrêt du programme par l\'utilisateur')
        
        break

    if key == ord('o'):
        
        ksize = (valeur, valeur)
    if key == ord('k'):
        i = i + 1
    if key == ord('l'):
        i = i - 1
    if key == ord('z'):
        cv2.imwrite('test.jpg', gradient)
        cv2.destroyWindow('apres transformation morphologique')
        print("successfully saved")
  

        

    
