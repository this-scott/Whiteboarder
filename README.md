# Whiteboarder Prototype

## Current State
* Disclaimer: I sketched this together in a day, this is an EXTREME PROTOTYPE
### Process
1. Calls a Mask-RCNN segmentation model trained on [this dataset](https://universe.roboflow.com/whiletrue-xopuj/whiteboard-semantic-seg)
2. Contour detects the mask to return an oriented bounding box surrounded the whiteboard
    * Steps can be replaced with a more traditional cv implementation based on feature and edge detection or a custom, lightweight, model. These implementations date back to 2004 using foundational methods that are layered on to create popular detection models.
    * Should also start position locking
3. Performs a rectification of the detected paper or whiteboard that expands 
    * Currently, the transformation performed does not account for whiteboard size and perspective and must be recreated. Right now it finds the largest differences between each parallel side and fits it to a rectangle

## Goals
1. Take an image
    * Image grabbing functionality would happen on an app but would occur 
1. Perspective transformation
    * Step 3 but working. 
    * Warp perspective to appear as if facing directly
2. Instant access
    * Uploaded files are accessible by signing in on a website
    * Landing on sign in is files because there's nothing else that needs to be accessed
    * Files last 15 minutes but can be to a permanent storage source(local, drive, etc.)(pdf or image format)
3. Feature Extraction
    * OCR on existing texts. Use Tesseract
    * Detect shapes, boxes, and drawings
    * Every item on the drawing should be associated with something
4. Integration Handling
    * Create pastable text extractions for quick notes integration
    * Perform natural language processing to create dates and todo lists integrateable to google tasks or an ics download
    * Shapes and diagrams can be converted into their own images or **Figma / Lucidchart / Canva objects** (this might be the big seller)
        * Poisson distrubutions to generate handwriting datasets

## Installation and Setup
1. `pip install requirements.txt`
2. Source the path to your models file

*Images coming soon*