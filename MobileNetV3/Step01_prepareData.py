import os
import random
import shutil

splitsize = .50
categories = []

source_folder = "C:/Users/USER/tensored-django-main/MobileNetV3/full_dataset_for_model_v3"
folders = os.listdir(source_folder)
print(folders)

for subfolder in folders:
    if os.path.isdir(source_folder + "/" + subfolder):
        categories.append(subfolder)

categories.sort()
print(categories)

target_folder = "C:/Users/USER/tensored-django-main/MobileNetV3/final_dataset_for_model_v3"
existDataSetPath = os.path.exists(target_folder)
if existDataSetPath==False:
    os.mkdir(target_folder)

# Create a function for split data for train and validation
    
def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files=[]

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
    print(len(files))

    trainingLength = int(len(files) * SPLIT_SIZE)
    shuffleSet = random.sample(files, len(files))
    trainingSet = shuffleSet[0:trainingLength]
    validSet = shuffleSet[trainingLength:]


    #copy the train images
    for filename in trainingSet:
        thisFile = SOURCE + filename
        destination = TRAINING + filename
        shutil.copyfile(thisFile, destination)
    
    #copy the validation images
    for filename in validSet:
        thisFile = SOURCE + filename
        destination = VALIDATION + filename
        shutil.copyfile(thisFile, destination)

trainPath = target_folder + "/train"
validatePath = target_folder + "/validate"

#create the target folders
existDataSetPath = os.path.exists(trainPath)
if existDataSetPath==False:
    os.mkdir(trainPath)

existDataSetPath = os.path.exists(validatePath)
if existDataSetPath==False:
    os.mkdir(validatePath)


#lets run the function for each of the folders
for category in categories:
    trainDestPath = trainPath + "/" + category
    validateDestPath = validatePath + "/" + category

    if os.path.exists(trainDestPath)==False:
        os.mkdir(trainDestPath)
    if os.path.exists(validateDestPath)==False:
        os.mkdir(validateDestPath)

    
    sourcePath = source_folder + "/" + category + "/"
    trainDestPath = trainDestPath + "/"
    validateDestPath = validateDestPath + "/"

    print("COpy from: " +sourcePath + " to: " + trainDestPath + " and: " + validateDestPath)

    split_data(sourcePath, trainDestPath, validateDestPath, splitsize)