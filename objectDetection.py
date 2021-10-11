import cv2


# objectDetection
# Parameters:   needle_image, image we search for
#               hay_image,    image we look in
# Return value: Center position as (x, y)

def objectDetection(needle_image, hay_image):

    needle_image = cv2.read(needle_image)
    hay_image = cv2.read(hay_image)

    width, height = hay_image.shape[::-1]

    method = cv2.TM_CCOEFF_NORMED

    match_result = cv2.matchTemplate(needle_image, hay_image, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)

    center = (max_loc[0] + width/2, max_loc[1] + height/2)

    return center




