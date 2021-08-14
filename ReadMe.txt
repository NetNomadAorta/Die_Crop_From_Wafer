To Run:
---------------------------------------
- You will need python version 3.6 and above to run, and have cv2 and imutils libraries installed.

- Place wafer sttiched-image in the "R:\public\Toolkits\Die_Crop_From_Wafer\Images\Images_to_Scan" folder 
	labeled as "Original.jpg". Crop a die or golden-image from this wafer image, then place this cropped 
	die in the "R:\public\Toolkits\Die_Crop_From_Wafer\Images\Images_to_Compare_for_Cropping" folder
	as "toCompare.jpg". Then run "die_cropper.py" in the main directory.

- After each picked up and cropped die found in the wafer, the program will place cropped dies/images in the 
	"R:\public\Toolkits\Die_Crop_From_Wafer\Images\Cropped_Images" folder for you to inspect!