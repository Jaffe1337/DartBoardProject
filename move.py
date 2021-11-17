import os
import shutil


testPath = r"C:\Users\janfr\OneDrive - Universitetet i Agder\Desktop\TensorFlow\workspace\training_demo\images\train"
trainedPath = r"C:\Users\janfr\OneDrive - Universitetet i Agder\Desktop\TensorFlow\workspace\training_demo\images\trained"


def moveFile(file):

    currentPath = trainedPath + "/" + file + ".xml"
    newPath = testPath + "/" + file + ".xml"

    shutil.move(currentPath, newPath)


for testFile in os.listdir(testPath):
    for trainFile in os.listdir(trainedPath):
        if trainFile[:-4:] == testFile[:-4:]:
            moveFile(trainFile[:-4:])
