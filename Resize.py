import cv2
import os

# C:\Users\janfr\ikt213\Project\TensorFlow\workspace\training_demo\images
imagePath = r"C:\Users\janfr\ikt213\Project\TensorFlow\workspace\training_demo\images\demo"
savePath = r"C:\Users\janfr\ikt213\Project\TensorFlow\workspace\training_demo\images\newdemo"


def resize(img):
    dim = (640, 640)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def save(img, filename):
    cv2.imwrite(savePath + "/" + filename, img)
    print(f"saved image to {savePath}/{filename}")


for file in os.listdir(imagePath):
    image = cv2.imread(imagePath + "/" + file)
    image = resize(image)
    save(image, file)

