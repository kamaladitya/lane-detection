import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import os
from PIL import Image
from moviepy.editor import VideoFileClip

def nothing(x):
    pass

def topview(image):
        height,width,col = image.shape
        src = np.array([[0,0],[width,0],[width,height],[0,height]],dtype='float32')
        dest = np.array([[0,0],[width,0],[height,479],[210,height]],dtype='float32')
        h, status = cv2.findHomography(src, dest)
        imgd = np.zeros((height,width,col),dtype='uint8')
        imgd = cv2.warpPerspective(image, h, (width,height))
        return imgd,h


def road_lines(original_image,diff,diff_prev):
	print("diff=",diff)
	height,width,_ =original_image.shape
	top_view,h = topview(original_image)
	red_chan=top_view[:,:,0]
	th, binary_warped = cv2.threshold(red_chan, 240, 255, cv2.THRESH_BINARY)
	image = cv2.medianBlur(binary_warped,11 )
	leftx=[]
	lefty=[]
	rightx=[]
	righty=[]
	histogram = np.sum(image[np.int(binary_warped.shape[0]/2):,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	midpoint=midpoint-diff
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	# Choose the number of sliding windows
	nwindows = 15
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
	margin = 50
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	nwind=0
	f=1
	fleft=1
	fright=1
	fno=1
	dist_cur=0
	dist_prev=0
       # Step through the windows one by one
	for window in range(nwindows):
		nwind=nwind+1
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
		if(len(good_right_inds)>minpix and len(good_left_inds)>minpix):
			print("right> left>")
			if(nwind==2):
				if(len(leftx)!=0 and len(rightx)!=0):
					midx=int(((sum(leftx)/len(leftx))+(sum(rightx)/len(rightx)))/2)
					diff=320-midx
					diff_prev=diff
					#midx=(leftx+rightx)/2
					#print(midx)
				else:
					diff=diff_prev
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			leftx.extend(nonzerox[good_left_inds])
			lefty.extend(nonzeroy[good_left_inds])
			rightx.extend(nonzerox[good_right_inds])
			righty.extend(nonzeroy[good_right_inds])
			flag=1
			dist_cur=np.mean(np.array(nonzerox[good_right_inds]))-np.mean(np.array(nonzerox[good_left_inds]))
		else:
			flag=0
			dist_cur=dist_prev
		if(flag==1):
			dist_prev=dist_cur
		if(len(good_left_inds)<minpix and len(good_right_inds)>minpix):
			print("left< right>")			
			fleft=0
			if(fright==0):
				print("fright break")
				break
			print(dist_cur)
			if(dist_cur!=0 and fno==1):
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
				leftx_current = np.int(np.mean(nonzerox[good_right_inds]-dist_cur))
				leftx.extend(nonzerox[good_right_inds]-int(dist_cur))
				lefty.extend(nonzeroy[good_right_inds])
				rightx.extend(nonzerox[good_right_inds])
				righty.extend(nonzeroy[good_right_inds])
				print("dist!=0",len(leftx))
			else:
				leftx.extend([20]*window_height)
				lefty.extend(list(range((nwind-1)*window_height,nwind*window_height)))
				rightx.extend(nonzerox[good_right_inds])
				righty.extend(nonzeroy[good_right_inds])
				print("dist=0",len(leftx))
		elif(len(good_right_inds)<minpix and len(good_left_inds)>minpix):
			print("left> right<")
			print("rigt pixel",len(good_right_inds))
			fright=0
			if(fleft==0):
				print("fleft break")
				break
			if(dist_cur!=0 and fno==1):
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
				rightx_current = np.int(np.mean(nonzerox[good_left_inds])+dist_cur)
				rightx.extend(nonzerox[good_left_inds]+int(dist_cur))
				righty.extend(nonzeroy[good_left_inds])
				leftx.extend(nonzerox[good_left_inds])
				lefty.extend(nonzeroy[good_left_inds])
				print("dist!=0",len(rightx))
			else:
				rightx.extend([450]*window_height)
				righty.extend(list(range((nwind-1)*window_height,nwind*window_height)))
				leftx.extend(nonzerox[good_left_inds])
				lefty.extend(nonzeroy[good_left_inds])
				print("dist=0",len(rightx))
		elif(len(good_left_inds)<minpix and len(good_right_inds)<minpix):
			print("left< right<")
			break
	print("end")
# Concatenate the arrays of indices
	leftx=np.array(leftx)
	lefty=np.array(lefty)
	rightx=np.array(rightx)
	righty=np.array(righty)
	left=np.vstack((leftx,lefty))
	right=np.vstack((rightx,righty))

# Fit a second order polynomial to each
	if(len(leftx)!=0 and len(lefty)!=0 and len(rightx)!=0 and len(righty)!=0):
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
	else:
		left_fit=[0,0,0]
		right_fit=[0,0,0]
# At this point, you're done! But here is how you can visualize the result as well:

# Generate x and y values for plotting
	ploty = np.linspace(binary_warped.shape[0],binary_warped.shape[0]-(nwind*window_height),binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	left=np.column_stack((np.array(left_fitx),ploty))
	left = left.reshape((-1,1,2))
	right=np.column_stack((np.array(right_fitx),ploty))
	right = right.reshape((-1,1,2))
	pts=np.concatenate((left,right),axis=0)

	mid=(right+left)/2
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	img = np.dstack((warp_zero, warp_zero, warp_zero))
	cv2.polylines(rgb_image, np.int32(right), isClosed=True,color=(240,0,0), thickness=15)
	cv2.polylines(rgb_image, np.int32(left), isClosed=True,color=(0,0,255), thickness=15)
	cv2.polylines(rgb_image, np.int32(mid), isClosed=True,color=(0,255,0), thickness=15)
	cv2.fillPoly(rgb_image,np.int32(pts) ,(255,0,0))
	height,width,col = rgb_image.shape
	src = np.array([[0,0],[width,0],[width,height],[0,height]],dtype='float32')
	dest = np.array([[0,0],[width,0],[height,479],[210,height]],dtype='float32')
	h, status = cv2.findHomography(dest,src)
	imgd = np.zeros((height,width,col),dtype='uint8')
	#print(imgd[:,1].shape)
	imgd = cv2.warpPerspective(rgb_image, h, (width,height))
	result=cv2.addWeighted(original_image,0.5,imgd,0.5,0)
	return result,diff,diff_prev
def binary_thres_red(original_image):
	height,width,_ = original_image.shape
	top_view,h = topview(original_image)
	red_chan=top_view[:,:,0]
	th, dst = cv2.threshold(red_chan, 220, 255, cv2.THRESH_BINARY)
	kernel=np.ones((5,5))
	#erosion=cv2.erode(dst,kernel,iterations=5)
	#dilation = cv2.dilate(erosion,kernel,iterations = 1)
	final = cv2.medianBlur(dst, 9)
	return(dst)
diff=0
diff_prev=0
a=1
video="5.avi"
if(a==0):
	vid_output = '5.mp4'
	video_output='output_thres_rednew.mp4'
	clip1 = VideoFileClip(video)
	vid_clip = clip1.fl_image(road_lines)
	vid_clip.write_videofile(vid_output, audio=False)
	vid_clip_red_thres = clip1.fl_image(binary_thres_red)
	vid_clip_red_thres.write_videofile(video_output, audio=False)
else:
	cap=cv2.VideoCapture(video)
	count=0
	while(True):
		ret,frame=cap.read()
		final,diff,diff_prev=road_lines(frame,diff,diff_prev)
		count=count+1
		print(count)
		cv2.imshow("frame",final)
		if(cv2.waitKey(1) and 0xFF==ord('q')):
			break
	cap.release()
	cv2.destroyAllWindows()
