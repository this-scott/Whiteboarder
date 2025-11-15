import argparse
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import os

@torch.no_grad()
def predict(model, image, device, threshold=0.5, draw_annotations=True):
    """
    Run inference on a single image.
    
    Args:
        model: Trained Mask R-CNN model
        image: PIL Image, numpy array (cv2), or tensor
        device: Device to run on
        threshold: Confidence threshold for predictions
        draw_annotations: Whether to draw annotations on image
    
    Returns:
        Dictionary with:
            - boxes: Bounding boxes
            - labels: Class labels
            - scores: Confidence scores
            - masks: Segmentation masks
            - annotated_image: Image with drawn annotations (if draw_annotations=True)
    """
    import cv2
    
    model.eval()
    
    # Store original image for annotation
    original_image = None

    #cv2 image handler
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Convert to tensor
    if isinstance(image, Image.Image):
        transform = T.ToTensor()
        image = transform(image)
    
    image = image.to(device)
    prediction = model([image])[0]
    
    # Filter by threshold
    keep = prediction['scores'] > threshold
    
    result = {
        'boxes': prediction['boxes'][keep].cpu().numpy(),
        'labels': prediction['labels'][keep].cpu().numpy(),
        'scores': prediction['scores'][keep].cpu().numpy(),
        'masks': prediction['masks'][keep].cpu().numpy()
    }
    
    # Draw annotations if requested
    annotated_image, contours = draw_predictions(
        original_image,
        result['boxes'],
        result['labels'],
        result['scores'],
        result['masks']
    )
    
    return result, annotated_image, contours


def draw_predictions(image, boxes, labels, scores, masks, alpha=0.5):
    """
    Draw bounding boxes, labels, scores, and masks on image.
    
    Args:
        image: BGR image (numpy array)
        boxes: Bounding boxes
        labels: Class labels
        scores: Confidence scores
        masks: Segmentation masks
        alpha: Transparency for mask overlay
    
    Returns:
        Image with annotations drawn
    """
    annotated = image.copy()
    approxes = []

    for i in range(len(boxes)):
        label = int(labels[i])
        if label != 1:
            continue
        box = boxes[i].astype(int)

        score = scores[i]
        mask = masks[i, 0]  # Shape: (H, W)

        color = (0,255,0)
        
        # """Getting cutout of bounding box, then rectangle from cv2 findcontours"""
        # cropped_boxes = []
        # for box in boxes:
        #     x1, y1, x2, y2 = box.astype(int)
        #     # Clip coordinates to image boundaries
        #     x1, y1 = max(0, x1), max(0, y1)
        #     x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        #     cropped = image[y1:y2, x1:x2].copy()
        #     cropped_boxes.append(cropped)

        # Draw mask
        mask_binary = (mask > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask_binary, (image.shape[1], image.shape[0]))
        
        contours, hierarchy = cv2.findContours(mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #break into 4 sided shapes
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)

            # Try increasing epsilon until you get 4 vertices
            for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]:
                epsilon = eps_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    approxes.append(approx)
                    break

            cv2.drawContours(annotated, approxes, -1, (0,0,255), 2)

        # Create colored mask
        # colored_mask = np.zeros_like(annotated)
        # colored_mask[mask_resized == 1] = color
        
        # Blend mask with image
        # annotated = cv2.addWeighted(annotated, 1, colored_mask, alpha, 0)
        
        # Draw bounding box
        # cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw label and score
        label_text = f"Class {label}: {score:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            annotated,
            (box[0], box[1] - text_height - baseline - 5),
            (box[0] + text_width, box[1]),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            label_text,
            (box[0], box[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return annotated, approxes

def get_maskrcnn_model(num_classes):
    """
    Build Mask R-CNN model with custom number of classes.
    
    Args:
        num_classes: Number of classes (including background)
    """
    # Load pretrained model
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

#TODO: Abandon this function
def rectify(corners):
    """Rectify generated mask of contours. 
    THIS DOES NOT ACCOUNT FOR PERSPECTIVE, INSTEAD IT RECTIFIES ACCORDING TO LARGEST DISTANCE BETWEEN TWO POINTS
    
    Args:
        corners (2d array)

    Returns:
        Mat: Transformation matrix to converting contour mask into a rectangle
        (int, int): Tuple representing size of rectified mask 
    """

    corners = corners.reshape(4, 2)

    #Assigning locations to each corner
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)

    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]   
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    (tl, tr, br, bl) = rect

    #get new height and width
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return M, (maxWidth, maxHeight)    

def main():
    """Main Function 
    Loads model, spawns Video Capture, and calls inference
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Capture Single Image")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    desired_width = 1260
    desired_height = 720

    # Load de model
    num_classes = 4  # 0: base 1: Whiteboard 2: No whiteboard(blame the dataset) 3: ?
    model = get_maskrcnn_model(num_classes)
    mp = os.environ.get("MODEL_PATH")
    
    model.load_state_dict(torch.load(mp))
    model.to(DEVICE)

    #capture loading 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    """MAIN LOOP"""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("couldn't grab frame")
            break
        
        annotations, image, contours = predict(model, frame, DEVICE)
        cv2.imshow('Camera Stream', image)

        #exit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            print(contours)
            M, nsize = rectify(contours[0]) # This doesn't work. This is considered a decomposition problem and trying to make a destination matrix isn't a solution.
            warped = cv2.warpPerspective(frame, M, nsize)
            cv2.imwrite('test.jpg',warped)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()