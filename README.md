# Boat detection to help mitigate the spread of AIS using Computer Vision.
Script built to detect boats in images to help track the movement of boats and therefore potential Aquatic Invasive Species hitching a ride. Can also process videos to count the number of boats crossing a waterway to help with the same objective.

### Installation
This was devolped with python 3.7

If you would like to do object training you will have to upgrade imageai

Precomputed CNN model needed for application (move to same folder as the image_processor.py script)

[YOLOV3](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5)

[RESNET](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
)

[YOLO-TINY](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5)


#### In command line(windows) terminal(mac)
navigate into the directory with requirements.txt and run

`pip install -r requirements.txt`

## image_processor.py

#### Inputs:
any image in the src_img folder
#### Outputs:
in processed_imgs folder then further split into folders containing the object that were detected in the images

### OPTIONS

#### COLOUR AND SPEED:

**must have at least one from each pair**

You need to pick a colour space and a model speed

colour space | speed
---|---
colour|fast
gray|normal


you can run the model at normal speed or fast 
this changes how the image is segmented to find boats and changes model characteristics

##### current cut offs for determining cut offs are:
 
*these are low but are doing decent considering we are using the stock CNN with no custom training*

Combo| Detection Limit
---|---
gray-fast|       5.86%
gray-normal|     8.17%
colour-fast |    5.98%
colour-normal|   7.28% (default)

####MODELS:

MODEL CODES  | Full name| Notes
---|---|---
y           | yoloV3 |(this model can now be trained but not implamented yet, training this using imageAI would be best moving forward)
yl          | yoloTinyV3| 
r           | resnet50 |(default and best for this applications currently )


####OBJECTS:
you can change what objects it will exclusively looks for my changing the custom_objects para

**common objects:**
* boat
* truck
* car
* ect.

## video_processor.py
#### Inputs:
only one video can go in the src_vid folder and it will process only the first file it finds
#### Outputs:
it will output video_out.avi in the same file directory as video_processor.py


### Parameters:
Optimized for 1 frame every 2 seconds (.5 fps) videos typically time lapses  (this is the fastest it can run at real time)

parameter | what it effects
---|---
mod| how far around the previous boat detected to look to identify if it is the same boat from the previous frame
frame scope| how many frames to check previous from current frame to determine if it is the same boat as seen before
model| pretty much has to be 'y' (yolo) only model that consistently detect boats on water the best
speed| normal and fast work best, changes how it segments the image to make the speed faster 

you can stream video from a camera some instructions exist in code
