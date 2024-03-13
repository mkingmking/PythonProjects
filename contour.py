import numpy as np
import cv2



image = cv2.imread('prj.png')  
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
#ret,thresh1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)



   

ret,thresh1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
cv2.waitKey()




#ret, thresh1 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(image, contours2, -1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('SIMPLE Approximation contours', image)
cv2.waitKey(0)

######versions differents de image contouring
"""
image_copy3 = image.copy()
for i, contour in enumerate(contours2): # loop over one contour area
   for j, contour_point in enumerate(contour): # loop over the points
       # draw a circle on the current contour coordinate
       cv2.circle(image_copy3, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 0), 2, cv2.LINE_AA)
# see the results
cv2.imshow('CHAIN_APPROX_SIMPLE Point only', image_copy3)

"""
"""
contours4, hierarchy4 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
image_copy5 = image.copy()
cv2.drawContours(image_copy5, contours4, -1, (0, 255, 0), 2, cv2.LINE_AA)
# see the results
cv2.imshow('EXTERNAL', image_copy5)
print(f"EXTERNAL: {hierarchy4}")


"""

'''
img_gray = cv2.medianBlur(img_gray, ksize=7)
#img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel, iterations = 1)

ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)



contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE , method=cv2.CHAIN_APPROX_NONE)

image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

cv2.imshow('None approximation', image_copy)



contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw contours on the original image for `CHAIN_APPROX_SIMPLE`
image_copy1 = image.copy()
cv2.drawContours(image_copy1, contours1, -1, (0, 255, 0), 2, cv2.LINE_AA)
# see the results
cv2.imshow('Simple approximation', image_copy1)
cv2.waitKey(0)


'''




