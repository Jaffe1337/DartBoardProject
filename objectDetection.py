import cv2


# objectDetection
# Parameters:   needle_image, image we search for
#               hay_image,    image we look in
# Return value: Center position as (x, y)

def objectDetection(needle_image, hay_image):

    # Define values
    needle_image = cv2.imread(needle_image)
    hay_image = cv2.imread(hay_image)

    width, height = needle_image.shape[1::-1]

    # Object detection of choice
    method = cv2.TM_CCOEFF_NORMED

    # Perform the image matching
    match_result = cv2.matchTemplate(needle_image, hay_image, method)

    # Get the position of the needle image in the hay image
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)

    # Return value in (int, int)
    center = (int(max_loc[0] + width/2), int(max_loc[1] + height/2))

    # Display dot on image for testing purposes: delete after testing
    image = cv2.circle(hay_image, (center[0], center[1]), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.imshow("Image", image)
    cv2.waitKey()
    ###

    return center




