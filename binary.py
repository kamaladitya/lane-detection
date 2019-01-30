import matplotlib.pyplot as plt
import cv2
import numpy as np


def topview(image):
        height,width,col = image.shape
        src = np.array([[0,0],[639,0],[639,479],[0,479]],dtype='float32')
        dest = np.array([[0,0],[639,0],[480,479],[210,height]],dtype='float32')
        h, status = cv2.findHomography(src, dest)
        imgd = np.zeros((height,width,col),dtype='uint8')
        imgd= cv2.warpPerspective(image, h, (width,height))
        return imgd,h
im=cv2.imread("frame0.jpg")
cv2.imshow("image",im)
cv2.waitKey(0)
height,width,_ = im.shape

im,h = topview(im)
cv2.imshow("image",im)
cv2.waitKey(0)
cv2.imwrite("3.jpeg",im)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)


fig,_axs = plt.subplots(nrows=4,ncols=3)
fig.subplots_adjust(hspace=0.3)
axs=_axs.flatten()

axs[0].imshow(im[:,:,0])
axs[0].set_title("Red")
axs[1].imshow(im[:,:,1])
axs[1].set_title("Green")
axs[2].imshow(im[:,:,2])
axs[2].set_title("Blue")
axs[3].imshow(ycrcb[:,:,0])
axs[3].set_title("YCrCb_Y")
axs[4].imshow(ycrcb[:,:,1])
axs[4].set_title("YCrCb_Cr")
axs[5].imshow(ycrcb[:,:,2])
axs[5].set_title("YCrCb_Cb")
axs[6].imshow(hsv[:,:,0])
axs[6].set_title("HSV_H")
axs[7].imshow(hsv[:,:,1])
axs[7].set_title("HSV_S")
axs[8].imshow(hsv[:,:,2])
axs[8].set_title("HSV_V")
axs[9].imshow(lab[:,:,0])
axs[9].set_title("LAB_L")
axs[10].imshow(lab[:,:,1])
axs[10].set_title("LAB_a")
axs[11].imshow(lab[:,:,2])
axs[11].set_title("LAB_b")

plt.show()

red_chan=im[:,:,0]
#print(red_chan)
thres=(200,255)
output=np.zeros_like(red_chan)
#output[(red_chan>=thres[0])&(red_chan<=thres[1])]=255
th, dst = cv2.threshold(red_chan, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("out",dst)
cv2.waitKey(0)
final = cv2.medianBlur(dst, 9)
cv2.imshow("median",final)
cv2.waitKey(0)
cv2.imwrite("2.jpeg",final)


