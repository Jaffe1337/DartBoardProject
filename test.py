import os

from objectDetection import objectDetection


def test():
    image_1 = "Test/Needle.PNG"
    image_2 = "Test/Hay.PNG"

    test_detection = objectDetection(image_1, image_2)


def fileTest():
    for file in os.listdir("Test"):
        print(file)


fileTest()
