# Import the necessary packages
import os
import glob
import imutils
# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,100000).__str__()
import cv2
import time
# TESTING SVD FROM NUMPY
import numpy as np

# User Parameters/Constants to Set
MATCH_CL = 0.80 # Minimum confidence level (CL) required to match golden-image to scanned image
SPLIT_MATCHES_CL =  0.98 # Splits MATCH_CL to SPLIT_MATCHES_CL (defects) to one folder, rest (no defects) other folder
STICHED_IMAGES_DIRECTORY = "Images/Stitched_Images/"
GOLDEN_IMAGES_DIRECTORY = "Images/Golden_Images/"
SLEEP_TIME = 0.0 # Time to sleep in seconds between each window step


def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours) ,int(mins), round(sec)))


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


# PYRAMID NOT IN USE - It scales down images to crop downscale for ML
def pyramid(fullImage, goldenImage, scale=1.5, minSize = (500, 500)):
    # Yield the original full image
    yield (fullImage, goldenImage)
    
    # Keep looping over the pyramid
    while True:
        # Compute the new dimensions of the image and resize it
        w1 = int(fullImage.shape[1] / scale)
        resizedFullImage = imutils.resize(fullImage, width = w1)
        w2 = int(goldenImage.shape[1] / scale)
        resizedCropImage = imutils.resize(goldenImage, width = w2)
        
        # If the resized image does not meet the supplied minimum
        #     size, then stop constructing the pyramid
        if resizedFullImage.shape[0] < minSize[1] or resizedFullImage.shape[1] < minSize[0] \
            or resizedCropImage.shape[0] < minSize[1] or resizedCropImage.shape[1] < minSize[0]:
            break
        
        # yield the next image in the pyramid
        yield (resizedFullImage, resizedCropImage)


def slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
    # Slides a window across the stitched-image
    for y in range(0, fullImage.shape[0], stepSizeY):
        for x in range(0, fullImage.shape[1], stepSizeX):
            # Yield the current window
            yield (x, y, fullImage[y:y + windowSize[1], x:x + windowSize[0]])


# Comparison scan window-image to golden-image
def getMatch(window, goldenImage, x, y):
    h1, w1, c1 = window.shape
    h2, w2, c2 = goldenImage.shape
    
    if c1 == c2 and h2 <= h1 and w2 <= w1:
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(window, goldenImage, method)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > MATCH_CL: 
            print("\nFOUND MATCH")
            print("max_val = ", max_val)
            print("Window Coordinates: x1:", x + max_loc[0], "y1:", y + max_loc[1], \
                  "x2:", x + max_loc[0] + w2, "y2:", y + max_loc[1] + h2)
            
            # Gets coordinates of cropped image
            return (max_loc[0], max_loc[1], max_loc[0] + w2, max_loc[1] + h2, max_val)
        
        else:
            return ("null", "null", "null", "null", "null")

# NOT IN USE
def compressImage(imageToCompress, k):
    print("processing...")
    img = imageToCompress # getting image from
    
    # splitting the array into three 2D array (svd only apply on 2D array)
    r = img[:,:,0]  # array for R
    g = img[:,:,1]  # array for G
    b = img[:,:,2] # array for B
    
    print("compressing...")
    
    # Calculating the svd components for all three arrays
    ur,sr,vr = np.linalg.svd(r, full_matrices=False)
    ug,sg,vg = np.linalg.svd(g, full_matrices=False)
    ub,sb,vb = np.linalg.svd(b, full_matrices=False)
    
    # Forming the compress image with reduced information
    # We are selecting only k singular values for each array to make image which will exclude some information from the 
    # image while image will be of same dimension
    
    # ur (mxk), diag(sr) (kxk) and vr (kxn) if image is off (mxn)
    # so let suppose we only selecting the k1 singular value from diag(sr) to form image
    
    rr = np.dot(ur[:,:k],np.dot(np.diag(sr[:k]), vr[:k,:]))
    rg = np.dot(ug[:,:k],np.dot(np.diag(sg[:k]), vg[:k,:]))
    rb = np.dot(ub[:,:k],np.dot(np.diag(sb[:k]), vb[:k,:]))
    
    print("arranging...")
    
    # Creating a array of zeroes; shape will be same as of image matrix
    rimg = np.zeros(img.shape)
    
    # Adding matrix for R, G & B in created array
    rimg[:,:,0] = rr
    rimg[:,:,1] = rg
    rimg[:,:,2] = rb
    
    # It will check if any value will be less than 0 will be converted to its absolute
    # and, if any value is greater than 255 than it will be converted to 255
    # because in image array of unit8 can only have value between 0 & 255
    for ind1, row in enumerate(rimg):
        for ind2, col in enumerate(row):
            for ind3, value in enumerate(col):
                if value < 0:
                    rimg[ind1,ind2,ind3] = abs(value)
                if value > 255:
                    rimg[ind1,ind2,ind3] = 255

    # converting the compress image array to uint8 type for further conversion into image object
    compressed_image = rimg.astype(np.uint8)
    
    # # Showing the compressed image in graph
    # plt.title("Image Name: "+imageToCompress+"\n")
    # plt.imshow(compressed_image)
    # plt.axis('off')
    # plt.show()
    
    # Uncomment below code if you want to save your compressed image to the file
    #compressed_image = Image.fromarray(compressed_image)
    #compressed_image.save("image_name.jpg")
    
    return compressed_image

# MAIN():
# =============================================================================
# Starting stopwatch to see how long process takes
start_time = time.time()

# Clears some of the screen for asthetics
print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

# Deletes contents in cropped- and split-image folders
deleteDirContents("./Images/Cropped_Die_Images/")
deleteDirContents("./Images/Splitted_Cropped_Die_Images/Not_Likely_Defects/")
deleteDirContents("./Images/Splitted_Cropped_Die_Images/Potential_Defects/")
deleteDirContents("./Images/Failing_Dies_Overlayed_on_Wafer_Image")
# deleteDirContents("./Images/SVD_Cropped_Die_Images/")

# Load the first of each stitched- and golden-images
fullImagePath = glob.glob(STICHED_IMAGES_DIRECTORY + "*")
fullImage = cv2.imread(fullImagePath[0])
goldenImagePath = glob.glob(GOLDEN_IMAGES_DIRECTORY + "*")
goldenImage = cv2.imread(goldenImagePath[0])

# Load Stitched-Image Path
# Main Stitched-Image directory
mainStitchDir = glob.glob(STICHED_IMAGES_DIRECTORY + "*")


# Runs through each slot file within the main file within stitched-image folder
for slotDir in glob.glob(mainStitchDir[0] + "/*"): 
    print("Starting", slotDir, "\n")

    # Parameter set
    winW = round(goldenImage.shape[1] * 1.5) # Scales window width with full image resolution
    # BELOW DEFAULT IS 1.5 CHANGE BACK IF NEEDED
    winH = round(goldenImage.shape[0] * 1.3) # Scales window height with full image resolution
    windowSize = (winW, winH)
    stepSizeX = round(winW / 2.95)
    stepSizeY = round(winH / 2.95)
    
    # Predefine next for loop's parameters 
    prev_y1 = stepSizeY * 9 # Number that prevents y = 0 = prev_y1
    prev_x1 = stepSizeX * 9
    rowNum = 0
    colNum = 0
    prev_matchedCL = 0
    # TESTING BELOW
    prev_BadX1 = 0
    prev_BadY1 = 0
    prev_BadX2 = 0
    prev_BadY2 = 0
    # BELOW IS FINE
    BadX1 = 0
    BadY1 = 0
    BadX2 = 0
    BadY2 = 0
    
    for fullImagePath in glob.glob(slotDir + "/*"):
        fullImage = cv2.imread(fullImagePath)
    
        # TESTING BELOW FOR SAVING FULL IMAGE WITH BAD DIE BOXES
        fullImageClone = fullImage.copy()
        
        # loop over the sliding window
        for (x, y, window) in slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            
            # Draw rectangle over sliding window for debugging and easier visual
            displayImage = fullImage.copy()
            cv2.rectangle(displayImage, (x, y), (x + winW, y + winH), (255, 0, 180), 40)
            # TESTING BELOW
            # Add rect to failing area already saved
            cv2.rectangle(displayImage, (BadX1, BadY1), (BadX2, BadY2), (0, 100, 255), 40)
            displayImageResize = cv2.resize(displayImage, (1800, round(fullImage.shape[0] / fullImage.shape[1] * 1800)))
            # cv2.imshow(str(slotDir), displayImageResize) # TOGGLE TO SHOW OR NOT
            cv2.waitKey(1)
            time.sleep(SLEEP_TIME) # sleep time in ms after each window step
            
            # Scans window for matched image
            # ==================================================================================
            # Scans window and grabs cropped image coordinates relative to window
            # Uses each golden image in the file if multiple part types are present
            for goldenImagePath in glob.glob(GOLDEN_IMAGES_DIRECTORY + "*"):
                goldenImage = cv2.imread(goldenImagePath)
                # Gets coordinates relative to window of matched dies within a Stitched-Image
                win_x1, win_y1, win_x2, win_y2, matchedCL = getMatch(window, goldenImage, x, y)
                
                # Saves cropped image and names with coordinates
                if win_x1 != "null":
                    # Turns cropped image coordinates relative to window to stitched-image coordinates
                    x1 = x + win_x1
                    y1 = y + win_y1
                    x2 = x + win_x2
                    y2 = y + win_y2
                    
                    # Makes sure same image does not get saved as different names
                    if y1 >= (prev_y1 + round(stepSizeY / 2.95)) or y1 <= (prev_y1 - round(stepSizeY / 2.95)):
                        rowNum += 1
                        colNum = 1
                        sameCol = False
                    else:
                        if x1 >= (prev_x1 + round(stepSizeX / 2.95)) or x1 <= (prev_x1 - round(stepSizeX / 2.95)):
                            colNum += 1
                            prev_matchedCL = 0
                            sameCol = False
                        else: 
                            sameCol = True
                    
                    # NEEDS A CHECK TO SEE IF FIRST X IN PREVIOUS Y-ROW IS THE SAME
                    #   IF IT ISN'T, THEN MAKE PREVIOUS FIRST X IN PREVIOUS ROW
                    #   HAVE A COLUMN_NUMBER += 1 AND DELETE OLD SAVE AND RESAVE
                    #   WITH NEW NAME
                    
                    # Puts 0 in front of single digit row nad column number
                    if rowNum < 10:
                        rZ = 0
                    else: 
                        rZ = ""
                    if colNum < 10:
                        cZ = 0
                    else: 
                        cZ = ""
                    
                    if (sameCol == False) or (sameCol == True and matchedCL > prev_matchedCL): 
                        # Gets cropped image and saves cropped image
                        croppedImage = window[win_y1:win_y2, win_x1:win_x2]
                        cv2.imwrite("./Images/Cropped_Die_Images/Row_{}{}-Col_{}{}.jpg".format(rZ, rowNum, cZ, colNum), croppedImage)
                        
                        # # # TESTING BELOW FOR SVD
                        # # TESTIG BELOW
                        # SVDImage = compressImage(croppedImage, 100)
                        # # croppedImage_Copy = croppedImage.copy()
                        # # # set blue and green channels to 0
                        # # r = croppedImage_Copy[:, :, 2]
                        
                        # # # Uses grayscale instead
                        # # g = cv2.cvtColor(croppedImage_Copy, cv2.COLOR_BGR2GRAY)
                        
                        # # # Actual SVD conversion part
                        # # U, s, V = np.linalg.svd(g)
                        # # num_components = 200
                        # # SVDImage = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) * np.matrix(V[:num_components, :])
                        # # croppedImage_Copy_Copy = croppedImage_Copy
                        # # croppedImage_Copy_Copy[:, :, 0] = 0
                        # # croppedImage_Copy_Copy[:, :, 1] = 0
                        # # croppedImage_Copy_Copy[:, :, 2] = SVDImage
                        
                        # cv2.imwrite("./Images/SVD_Cropped_Die_Images/Row_{}{}-Col_{}{}.jpg".format(rZ, rowNum, cZ, colNum), SVDImage)
                        
                        # SAVES CROPPED TO NEW FOLDER WITH CROP NAME
                        os.makedirs("./Images/" + slotDir[23:-3] + "/" + \
                            slotDir[-2:] + "/Cropped_Die_Images", exist_ok=True)
                        cv2.imwrite("./Images/" + slotDir[23:-3] + "/" + \
                            slotDir[-2:] + "/Cropped_Die_Images" +\
                            "/Row_{}{}-Col_{}{}.jpg".format(rZ, rowNum, cZ, colNum), croppedImage)
                        
                        # Splits cropped images to folders with potential defects and no defects
                        if matchedCL > SPLIT_MATCHES_CL:
                            cv2.imwrite("./Images/Splitted_Cropped_Die_Images/Not_Likely_Defects/R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum, round(matchedCL * 100)), croppedImage)
                            # Saves CI labeled cropped image in newly created folder
                            os.makedirs("./Images/" + slotDir[23:-3] + "/" + \
                                slotDir[-2:] + "/Splitted_Cropped_Die_Images/Not_Likely_Defects", exist_ok=True)
                            cv2.imwrite("./Images/" + slotDir[23:-3] + "/" + \
                                slotDir[-2:] + "/Splitted_Cropped_Die_Images/Not_Likely_Defects" +\
                                "/R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum, round(matchedCL * 100)), croppedImage)
                            
                        else: 
                            # TESTING BELOW
                            BadX1 = x1
                            BadY1 = y1
                            BadX2 = x2
                            BadY2 = y2
                            cv2.imwrite("./Images/Splitted_Cropped_Die_Images/Potential_Defects/R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum,  round(matchedCL * 100)), croppedImage)
                            # Saves CI labeled cropped image in newly created folder
                            os.makedirs("./Images/" + slotDir[23:-3] + "/" + \
                                slotDir[-2:] + "/Splitted_Cropped_Die_Images/Potential_Defects", exist_ok=True)
                            cv2.imwrite("./Images/" + slotDir[23:-3] + "/" + \
                                slotDir[-2:] + "/Splitted_Cropped_Die_Images/Potential_Defects" +\
                                "/R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum, round(matchedCL * 100)), croppedImage)
                            
                        # If previous same Row and Column will be saved twice, deletes first one
                        if sameCol == True and matchedCL > prev_matchedCL:
                            fullImageClone = fullImageClone_Backup
                            BadX1 = 0
                            BadY1 = 0
                            BadX2 = 0
                            BadY2 = 0
                            if "R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum, round(prev_matchedCL * 100)) in os.listdir("./Images/Splitted_Cropped_Die_Images/Not_Likely_Defects/"): 
                                os.remove("./Images/Splitted_Cropped_Die_Images/Not_Likely_Defects/R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum, round(prev_matchedCL * 100)))
                                os.remove("./Images/" + slotDir[23:-3] + "/" + \
                                slotDir[-2:] + "/Splitted_Cropped_Die_Images/Not_Likely_Defects" +\
                                "/R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum, round(prev_matchedCL * 100)))
                            if "R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum, round(prev_matchedCL * 100)) in os.listdir("./Images/Splitted_Cropped_Die_Images/Potential_Defects/"): 
                                os.remove("./Images/Splitted_Cropped_Die_Images/Potential_Defects/R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum, round(prev_matchedCL * 100)))
                                os.remove("./Images/" + slotDir[23:-3] + "/" + \
                                slotDir[-2:] + "/Splitted_Cropped_Die_Images/Potential_Defects" +\
                                "/R{}{}-C{}{}-CL{}.jpg".format(rZ, rowNum, cZ, colNum, round(prev_matchedCL * 100)))
                    
                    prev_y1 = y1
                    prev_x1 = x1
                    prev_matchedCL = matchedCL
                    
                    # Draws orange boxes around bad dies
                    # Separate copy of resized full image with all bad dies showing orange boxes
                    # TESTING BELOW
                    # Saves backup of fullImageClone before next rectangle write
                    fullImageClone_Backup = fullImageClone.copy()
                    # Add rect to failing area already saved
                    cv2.rectangle(fullImageClone, (BadX1, BadY1), (BadX2, BadY2), (0, 50, 255), 35)
                    prev_BadX1 = BadX1
                    prev_BadY1 = BadY1
                    prev_BadX2 = BadX2
                    prev_BadY2 = BadY2
                # ==================================================================================
        rowNum += 1
        colNum = 0
        sameCol = False
        
        # Saves window with orange boxes around potential bad dies



        # cv2.imwrite("./Images/Failing_Dies_Overlayed_on_Wafer_Image/ImageWithBoxes.jpg", fullImageClone)
        # # TESTING BELOW
        # # SAVES Overlay Window TO NEW FOLDER WITH Stitched-Image NAME
        # os.makedirs("./Images/" + slotDir[23:-3] + "/" + \
        #     slotDir[-2:] + "/Failing_Dies_Overlayed_on_Wafer_Image", exist_ok=True)
        # cv2.imwrite("./Images/" + slotDir[23:-3] + "/" + \
        #     slotDir[-2:] + "/Failing_Dies_Overlayed_on_Wafer_Image" +\
        #     "/ImageWithBoxes.jpg", fullImageClone)

# Starting stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)