import json
import cv2
import numpy as np
import pycocotools.mask as mask

# Load 'things' and 'stuff' annotations
with open('coco_ann2017/annotations/instances_train2017.json', 'r') as f:
    things = json.load(f)

with open('stuff_trainval2017/stuff_train2017.json', 'r') as f:
    stuff = json.load(f)

# Function to convert binary mask to polygon
def polygon_from_mask(masked_arr):
    contours, _ = cv2.findContours(masked_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        if contour.size >= 6:  # Ensure valid polygons (at least 3 points)
            segmentation.append(contour.flatten().tolist())
    return segmentation

# Combine categories from 'stuff' into 'things', ensuring no duplicates
things_category_ids = {category['id'] for category in things['categories']}
for category in stuff['categories']:
    if category['id'] not in things_category_ids:
        things['categories'].append(category)

# Process each annotation in stuff
for ann in stuff['annotations']:
    if isinstance(ann['segmentation'], dict):  # If the segmentation is in RLE format
        rle_segmentation = ann['segmentation']
        height, width = rle_segmentation['size']
        
        # Decode the RLE to a binary mask
        masked_arr = mask.decode(rle_segmentation)
        
        # Convert the binary mask to polygons
        polygons = polygon_from_mask(masked_arr)
        
        # Add the polygons to the annotation
        ann['segmentation'] = polygons
        
        # Append this annotation to the things JSON data
        things['annotations'].append(ann)

# Save the modified 'things' data with the updated segmentations and categories into a new JSON file
with open('combined_things_stuff_train.json', 'w') as f:
    json.dump(things, f, indent=4)

print("Combined JSON file with polygon segmentations and merged categories has been created: 'combined_things_stuff.json'")
