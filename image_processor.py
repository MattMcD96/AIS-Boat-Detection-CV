from imageai.Detection import ObjectDetection
import time
import os
import imageio
import cv2
import numpy as np

# global setups
execution_path = os.getcwd()
detector = ObjectDetection()

# if you have a mask you have to set it on the line below and
# make sure you only use the imgs for that mask in the src_img folder and the mask is in the same folder as this scrit
usingMask = False

if usingMask:
    maskPath = 'deacon-msk.jpg'

'''
Options
-------
COLOUR AND SPEED:
**must have atleast one from each pair**
You need to pick a frame colour space and a model speed
colour space is colour(rgb) and gray

you can run the model at normal speed or fast 
this changes show the image is segmented to find id boats and model characteristics)

current cut offs for determining cut offs are: 
(these are low but are doing decent considering we are using the stock CNN with no custom training)
gray-fast       5.86%
gray-normal     8.17%
colour-fast     5.98%
colour-normal   7.28% (default)

MODELS:
MODEL CODES   Full name
y           | yoloV3 (this model can now be trained but not implamented yet, training this using imageAI would be best moving forward)
yl          | yoloTinyV3
r           | resnet50 (default and best for this applications currently )

OBJECTS:
change the custom_objects para
common objects 
boat (should always be set, should be only one set)
truck
car
ect.
'''

#MUST HAVE ATLEAST ONE FROM EACH PAIR
fast = False
normal = True

gray = False
colour = True

model = 'r'
custom_objects = detector.CustomObjects(boat=True)

# makes dirs for output pictures
if not os.path.exists(".\\processed_imgs\\vechicle_and_boats\\"):
    os.makedirs(".\\processed_imgs\\vechicle_and_boats\\")

if not os.path.exists(".\\processed_imgs\\boat\\"):
    os.makedirs(".\\processed_imgs\\boat\\")

if not os.path.exists(".\\processed_imgs\\vehicle\\"):
    os.makedirs(".\\processed_imgs\\vehicle\\")

if not os.path.exists(".\\processed_imgs\\nothing\\"):
    os.makedirs(".\\processed_imgs\\nothing\\")


# Deinfes the model with the tag provided above
def defmodel(speed):
    if model == 'r':
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
    elif model == 'y':
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    elif model == 'yl':
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath(os.path.join(execution_path, "yolo-tiny.h5"))
    detector.loadModel(detection_speed=str(speed))


def main():
    numboats = 0
    numVehicle = 0
    imgcount = 0
    total = 0

    if usingMask:
        mask = cv2.imread(str(maskPath))
        mask = mask / 255
    print('Processing Started')

    for filename in os.listdir(execution_path + '\src_img'):
        boat = False
        vehicle = False
        total += 1
        input_image = filename
        locn = input_image.split('.')
        imgcount += 1
        img_in = str(execution_path) + '\src_img\\' + str(input_image)
        img = cv2.imread(img_in)
        if usingMask:
            img = np.multiply(img, mask)
            img = img.astype(int)
            img = img.astype(np.uint8)

        cv2.imwrite('tempc.png', img)
        retimg3 = img
        retimg1 = img

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('tempg.png', img)
        retimg2=img
        retimg4=img

        if colour and fast:
            defmodel('fast')
            retimg1, detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                                                                        input_image="tempc.png",
                                                                        output_type="array",
                                                                        minimum_percentage_probability=5.98)

            for eachObject in detections:
                if eachObject["name"] == "boat":
                    boat = True
                    numboats += 1
                    print(str(input_image) + ',' + str(eachObject['percentage_probability']) + ',' + str(
                        eachObject['box_points']))

                if eachObject["name"] == "car":
                    vehicle = True
                    numVehicle += 1

                if eachObject["name"] == "truck":
                    vehicle = True
                    numVehicle += 1

        if gray and fast:
            defmodel('fast')
            retimg2, detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                                                                        input_image="tempg.png",
                                                                        output_type="array",
                                                                        minimum_percentage_probability=5.86)

            for eachObject in detections:
                if eachObject["name"] == "boat":
                    print(str(input_image) + ',' + str(eachObject['percentage_probability']) + ',' + str(
                        eachObject['box_points']))
                    boat = True

                if eachObject["name"] == "car":
                    vehicle = True
                    numVehicle += 1

                if eachObject["name"] == "truck":
                    vehicle = True
                    numVehicle += 1

        if colour and normal:
            defmodel("normal")

            retimg3, detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                                                                        input_image="tempc.png",
                                                                        output_type="array",
                                                                        minimum_percentage_probability=7.28)

            for eachObject in detections:
                if eachObject["name"] == "boat":
                    boat = True
                    numboats += 1
                    print(str(input_image) + ',' + str(eachObject['percentage_probability']) + ',' + str(
                        eachObject['box_points']))

                if eachObject["name"] == "car":
                    vehicle = True
                    numVehicle += 1

                if eachObject["name"] == "truck":
                    vehicle = True
                    numVehicle += 1

        if gray and normal:
            defmodel("normal")

            retimg4, detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                                                                        input_image="tempg.png",
                                                                        output_type="array",
                                                                        minimum_percentage_probability=8.17)

            for eachObject in detections:
                if eachObject["name"] == "boat":
                    boat = True
                    print(str(input_image) + ',' + str(eachObject['percentage_probability']) + ',' + str(
                        eachObject['box_points']))
                    numboats += 1

                if eachObject["name"] == "car":
                    vehicle = True
                    numVehicle += 1

                if eachObject["name"] == "truck":
                    vehicle = True
                    numVehicle += 1

        if colour and not gray:

            if vehicle and boat:
                imageio.imwrite('.\\processed_imgs\\vechicle_and_boats\\' + str(locn[0]) + '_det.jpg',
                                cv2.addWeighted(cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5,
                                                cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5, 0))
            elif boat:
                imageio.imwrite('.\\processed_imgs\\boat\\' + str(locn[0]) + '_det.jpg',
                                cv2.addWeighted(cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5,
                                                cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5, 0))
            elif vehicle:
                imageio.imwrite('.\\processed_imgs\\vehicle\\' + str(locn[0]) + '_det.jpg',
                                cv2.addWeighted(cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5,
                                                cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5, 0))

            else:
                imageio.imwrite('.\\processed_imgs\\nothing\\' + str(locn[0]) + '_det.jpg',
                                cv2.addWeighted(cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5,
                                                cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5, 0))

        if gray:
            retimg1 = cv2.cvtColor(retimg1, cv2.COLOR_RGB2GRAY)
            retimg3 = cv2.cvtColor(retimg3, cv2.COLOR_RGB2GRAY)
            if vehicle and boat:
                imageio.imwrite('.\\processed_imgs\\vechicle_and_boats\\' + str(locn[0]) + '_det.jpg',
                                cv2.addWeighted(cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5,
                                                cv2.addWeighted(retimg2, 0.55, retimg4, 0.55, 0), 0.5, 0))
            elif boat:
                imageio.imwrite('.\\processed_imgs\\boat\\' + str(locn[0]) + '_det.jpg',
                                cv2.addWeighted(cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5,
                                                cv2.addWeighted(retimg2, 0.55, retimg4, 0.55, 0), 0.5, 0))
            elif vehicle:
                imageio.imwrite('.\\processed_imgs\\vehicle\\' + str(locn[0]) + '_det.jpg',
                                cv2.addWeighted(cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5,
                                                cv2.addWeighted(retimg2, 0.55, retimg4, 0.55, 0), 0.5, 0))
            else:
                imageio.imwrite('.\\processed_imgs\\nothing\\' + str(locn[0]) + '_det.jpg',
                                cv2.addWeighted(cv2.addWeighted(retimg1, 0.55, retimg3, 0.55, 0), 0.5,
                                                cv2.addWeighted(retimg2, 0.55, retimg4, 0.55, 0), 0.5, 0))

    print('#-------Summary--------#')
    print('Number of images samples: ' + str(imgcount))
    print('Number of boats seen: ' + str(numboats))
    print('Number of vehicle seen: ' + str(numVehicle))


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))

if (not colour or not gray) and (not fast or not normal):
    print("ERROR: you must make sure you selected one of colour or gray AND normal or fast")

#clean up
if os.path.exists("tempg.png"):
  os.remove("tempg.png")

if os.path.exists("tempc.png"):
    os.remove("tempc.png")
