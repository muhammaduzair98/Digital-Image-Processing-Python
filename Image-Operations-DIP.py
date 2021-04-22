#Negative of Image with histograms

import cv2 
import matplotlib.pyplot as plt 
import numpy as np

# Read an image 
img_bgr = cv2.imread('/Users/eapple/moon.tif', 1)
plt.imshow(img_bgr) 
plt.show() 
color = ('b', 'g', 'r') 

for i, col in enumerate(color): 

    histrogram = cv2.calcHist([img_bgr], 
                        [i], None, 
                        [256], 
                        [0, 256]) 

    plt.plot(histrogram, color = col) 

    plt.xlim([0, 256]) 
    
plt.show() 

height, width, _ = img_bgr.shape 

for i in range(0, height - 1): 
    for j in range(0, width - 1): 

        pixel = img_bgr[i, j] 
        pixel[0] = 255 - pixel[0] 

        pixel[1] = 255 - pixel[1] 
        pixel[2] = 255 - pixel[2] 
    img_bgr[i, j] = pixel 

plt.imshow(img_bgr) 
plt.show() 

color = ('b', 'g', 'r') 

for i, col in enumerate(color): 

    histrogram = cv2.calcHist([img_bgr], 
                        [i], None, 
                        [256], 
                        [0, 256]) 

    plt.plot(histrogram, color = col) 
    plt.xlim([0, 256]) 

plt.show() 


#thresholds this image

import cv2
import numpy as np
import matplotlib.pyplot as plt
img= cv2.imread('/Users/eapple/moon.tif')
img1=cv2.imread('/Users/eapple/moon2.tif')
images=np.concatenate(img(img,img1),axis=1)
cv2.imshow("Images",images)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

hist=cv2.calcHist(gray_img,[0],None,[256],[0,256])
hist1=cv2.calcHist(gray_img1,[0],None,[256],[0,256])
plt.subplot(121)
plt.title("Image1")
plt.xlabel('bins')
plt.ylabel("No of pixels")
plt.plot(hist)
plt.subplot(122)
plt.title("Image2")
plt.xlabel('bins')
plt.ylabel("No of pixels")
plt.plot(hist1)
plt.show()

gray_img_eqhist=cv2.equalizeHist(gray_img)
gray_img1_eqhist=cv2.equalizeHist(gray_img1)
hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
hist1=cv2.calcHist(gray_img1_eqhist,[0],None,[256],[0,256])
plt.subplot(121)
plt.plot(hist)
plt.subplot(122)
plt.plot(hist1)
plt.show()


eqhist_images=np.concatenate((gray_img_eqhist,gray_img1_eqhist),axis=1)
cv2.imshow("Images",eqhist_images)
cv2.waitKey(0)
cv2.destroyAllWindows()

clahe=cv2.createCLAHE(clipLimit=40)
gray_img_clahe=clahe.apply(gray_img_eqhist)
gray_img1_clahe=clahe.apply(gray_img1_eqhist)
images=np.concatenate((gray_img_clahe,gray_img1_clahe),axis=1)
cv2.imshow("Images",images)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Contrast Stretching

import cv2
import numpy as np

img = cv2.imread('/Users/eapple/moon.tif')
original = img.copy()
xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
img = cv2.LUT(img, table)
cv2.imshow("original", original)
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows() 

#Flipping Image 
import cv2
  
originalImage = cv2.imread('/Users/eapple/moon.tif')
  
flipVertical = cv2.flip(originalImage, 0)
flipHorizontal = cv2.flip(originalImage, 1)
flipBoth = cv2.flip(originalImage, -1)
 
cv2.imshow('Original image', originalImage)
cv2.imshow('Flipped vertical image', flipVertical)
cv2.imshow('Flipped horizontal image', flipHorizontal)
cv2.imshow('Flipped both image', flipBoth)
 
 
cv2.waitKey(0)
cv2.destroyAllWindows()

