# import the necessary packages
import imutils
import cv2, glob, time, argparse, json

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
    
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
        
		# if the resized image does not meet the supplied minimum
		#     size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
        
		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize): # stepSize normally 4 to 8 (pixels)
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



# Comparison Scan
# ======================
def get_match(window, crops, x, y):
    print("\n")
    print("get_match started\n")
    h1, w1, c1 = window.shape
    crop_img = cv2.imread(crops)
    h2, w2, c2 = crop_img.shape
    if c1==c2 and h2<=h1 and w2<=w1:
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(window,crop_img,method)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val>0.80: # ORIGINALLY: 0.98
            print("FOUND MATCH")
            print("max_val = ", max_val) # DELEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEETE
            print("Coordinates: x1:", max_loc[0], "y1:", max_loc[1], "x2:", max_loc[0]+w2, "y2:", max_loc[1]+h2)
#            print("max_val",max_val)
# <-- DELETE COMMENT            print(max_loc[0],max_loc[1],max_loc[0]+w2,max_loc[1]+h2)
            # Gets coordinates of cropped image
            yield max_loc[0]
            yield max_loc[1]
            yield max_loc[0]+w2
            yield max_loc[1]+h2 # Gets coordinates of cropped image
        else:
            yield "null"
            yield "null"
            yield "null"
            yield "null"