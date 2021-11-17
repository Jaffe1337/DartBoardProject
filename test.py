import os

from objectDetection import objectDetection


def test():
    image_1 = "Test/Needle.PNG"
    image_2 = "Test/Hay.PNG"

    test_detection = objectDetection(image_1, image_2)


def fileTest():
    new_list = []
    for file in os.listdir(r"C:\Users\janfr\ikt213\Project\TensorFlow\workspace\training_demo\images\demo"):
        new_list.append(file)
    print(new_list)


fileTest()
