import numpy as np
import cv2
import matplotlib.pyplot as plt

def direction_vector(midx,midy):
	dvx=[]
	dvy=[]
	for i in range(len(midy)-1,1,-1):
		dvx.append(midx[i-1]-midx[i])
		dvy.append(midy[i]-midy[i-1])
	return(dvx,dvy)


image=cv2.imread("2.jpeg")#binary_warped rgb
original_image=cv2.imread("frame0.jpg")
#im=cv2.imread("2.jpeg",0)
red_chan=image[:,:,0]
th, binary_warped = cv2.threshold(red_chan, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("binary_img",binary_warped)
cv2.waitKey()


histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    else:
        break    
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    else:
        break

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 
left=np.vstack((leftx,lefty))
right=np.vstack((rightx,righty))

if (len(leftx)==0 and len(lefty)==0):
	leftx=[0]*229
	lefty=list(range(229))
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# At this point, you're done! But here is how you can visualize the result as well:

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
left=np.column_stack((np.array(left_fitx),ploty))
left = left.reshape((-1,1,2))
right=np.column_stack((np.array(right_fitx),ploty))
right = right.reshape((-1,1,2))
pts=np.concatenate((left,right),axis=0)


diffx=left-right
mid=(right+left)/2
midx=mid[:,:,0]
midy=mid[:,:,1]
dvx,dvy=direction_vector(midx,midy)
print("x=",np.mean(np.array(dvx)))
print("y=",np.mean(np.array(dvy)))
#print(np.array(dvx))

y_eval=200
left_curverad = np.absolute(((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2) ** 1.5)/(2 * left_fit[0]))
right_curverad = np.absolute(((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2) ** 1.5)/(2 * right_fit[0]))

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
cv2.imshow("out",out_img)
cv2.waitKey(0)

warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
img = np.dstack((warp_zero, warp_zero, warp_zero))
cv2.polylines(image, np.int32(right), isClosed=True,color=(255,0,0), thickness=15)
cv2.polylines(image, np.int32(left), isClosed=True,color=(0,0,255), thickness=15)
cv2.polylines(image, np.int32(mid), isClosed=True,color=(0,255,0), thickness=15)
cv2.fillPoly(image,np.int32(pts) ,(255,0,0))

#img=cv2.polylines(image,pts, False ,(255,0,0))
cv2.imshow("img",image)
cv2.waitKey(0)
height,width,col = image.shape
src = np.array([[0,0],[639,0],[639,479],[0,479]],dtype='float32')
dest = np.array([[0,0],[639,0],[480,479],[210,height]],dtype='float32')
h, status = cv2.findHomography(dest,src)
imgd = np.zeros((height,width,col),dtype='uint8')
#print(imgd[:,1].shape)
imgd[:,:,0] = cv2.warpPerspective(image[:,:,0], h, (width,height))
imgd[:,:,1] = cv2.warpPerspective(image[:,:,1], h, (width,height))
imgd[:,:,2] = cv2.warpPerspective(image[:,:,2], h, (width,height))


result=cv2.addWeighted(original_image,0.5,imgd,0.5,0)
print(original_image.shape)
#result=cv2.add(original_image,imgd)
cv2.imshow("final",result)
cv2.waitKey(0)









