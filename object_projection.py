# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:10:01 2024

@author: win
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------- function for testing and debugging ---------------- #

def plot_image(final_image, label="Final image"):
    """
    Plots the final image
    
    Parameters:
    final_image (numpy array): The field image with the projected object.
    projected_bbox_shape (list of tuples): Coordinates of the projected bounding box.
    """
    # Convert BGR (OpenCV format) to RGB for matplotlib display
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    
    # Create two subplots: one for the image, and one for the image with the bounding box
    
    # Plot the final image
    plt.imshow(final_image_rgb)
    plt.title(label)
    plt.axis('off')
    

    plt.show()
    
def plot_image_with_bbox(final_image, projected_bbox_shape):
    """
    Plots the final image with the projected bounding box.
    
    Parameters:
    final_image (numpy array): The field image with the projected object.
    projected_bbox_shape (list of tuples): Coordinates of the projected bounding box.
    """
    # Convert BGR (OpenCV format) to RGB for matplotlib display
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    
    # Draw the bounding box on the image before displaying
    projected_bbox_shape = np.array(projected_bbox_shape, dtype=np.int32)
    
    # Draw the bounding box on the image (modifies final_image_rgb in-place)
    pts = projected_bbox_shape.reshape((-1, 1, 2))
    cv2.polylines(final_image_rgb, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # Plot the final image with the bounding box
    plt.imshow(final_image_rgb)
    plt.title('Final Image with Bounding Box')
    plt.axis('off')

    plt.show()
    
# --------------------------------------------------------------------- # 


def resize(img, res_factor):
    
    height, width = img.shape[:2]
    
    res_width = int(width*res_factor)
    res_height = int(height*res_factor)
    resize_img = cv2.resize(img, (res_width, res_height ))
    
    return resize_img

def rotate_shape_and_bbox(object_img, bbox_shape):
    """
    Rotate the shape and corresponding bounding boxes by a random angle without clipping.
    
    Parameters:
    object_img (numpy array): The shape image (RGBA).
    bbox_shape (list of tuples): Bounding box coordinates for the shape.

    Returns:
    rotated_shape (numpy array): Rotated shape image with the alpha channel.
    rotated_bbox_shape (list of tuples): Rotated bounding box.
    """
    # Get the original dimensions of the image
    (h, w) = object_img.shape[:2]
    
    # Random rotation angle
    angle = random.uniform(0, 360)
    
    center = (w // 2, h // 2)
    M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Update the boundry box to fit the the new rotated image 
    cos = np.abs(M_rotate[0, 0])
    sin = np.abs(M_rotate[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to take into account translation
    M_rotate[0, 2] += (new_w / 2) - center[0]
    M_rotate[1, 2] += (new_h / 2) - center[1]
    
    # Expand the canvas to prevent clipping and rotate the entire image (RGBA)
    rotated_shape = cv2.warpAffine(object_img, M_rotate, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Rotate the bounding box points
    bbox_shape_rotated = cv2.transform(np.float32([bbox_shape]).reshape(-1, 1, 2), M_rotate).reshape(-1, 2)
    
    # Ensure alpha channel (transparency) is handled for interpolation
    # Split the rotated image into RGB and alpha channels
    rotated_rgb = rotated_shape[:, :, :3]
    rotated_alpha = rotated_shape[:, :, 3]
    
    # Combine the RGB and smoothed alpha channels
    final_image = cv2.merge([rotated_rgb[:, :, 0], rotated_rgb[:, :, 1], rotated_rgb[:, :, 2], rotated_alpha])
    
    return final_image, bbox_shape_rotated




def resize_dynamically(object_img, aerial_img, real_object_widty, altitude, real_ground_width):
    """
    Resize the object based on based on the aerial image altitude and real width in meter

    Parameters:
    object_img (numpy array): The input image to resize.
    object_real_size(float): the width of the object in meter.
    altitude(float): real aerial image altitude in meter.
    real_ground_width (float): real aeril image width in meter.

    Returns:
    resize_img (numpy array): The resized image.
    """
    
    obj_h, obj_w = object_img.shape[:2]
    aerial_h, aerial_w = aerial_img.shape[:2]
    
    
    # Calc the Ground Sampling Distance
    gsd = real_ground_width / aerial_w
    
    object_in_pixel = real_object_widty/gsd
    
    # Calc the scale factor and keep the aspect ratio
    scale_factor = object_in_pixel / obj_w  
    new_width = int(obj_w * scale_factor)
    new_height = int(obj_h * scale_factor)
    
    # Resize the object using OpenCV
    resized_object = cv2.resize(object_img, (new_width, new_height))

    return resized_object


def apply_super_resolution(obj_img):
    """
    Applies super-resolution to an image while preserving the alpha channel if it exists.

    Parameters:
    obj_img (numpy array): Input image, with or without an alpha channel.

    Returns:
    high_res_obj_img (numpy array): High-resolution image with the original alpha channel if it exists.
    """
    
    # Check if the input image has 4 channels (including alpha)
    has_alpha = obj_img.shape[2] == 4

    if has_alpha:  # CV_8UC4 format
        print("Converting from 4-channel (CV_8UC4) to 3-channel (CV_8UC3)")
        # Split the image into the BGR and Alpha channels
        bgr_img = cv2.cvtColor(obj_img, cv2.COLOR_BGRA2BGR)  # Convert to BGR (3 channels)
        alpha_channel = obj_img[:, :, 3]  # Extract the alpha channel
    else:
        bgr_img = obj_img  # If no alpha, process as usual

    # Initialize super-resolution model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    # Load the pre-trained super-resolution model
    sr.readModel("D:/Alex Eagles/ODLC/sencatic data generation/pretrained model/ESPCN_x4.pb")
    
    # Set the super-resolution model (ESPCN) and scale factor (x4)
    sr.setModel("espcn", 4)
    
    # Apply super-resolution to the BGR image
    high_res_bgr_img = sr.upsample(bgr_img)

    # If the original image had an alpha channel, resize and merge it back
    if has_alpha:
        # Resize the alpha channel to match the high-resolution output
        high_res_alpha_channel = cv2.resize(alpha_channel, (high_res_bgr_img.shape[1], high_res_bgr_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Merge the BGR image and the resized alpha channel back into a 4-channel (BGRA) image
        high_res_obj_img = cv2.merge((high_res_bgr_img, high_res_alpha_channel))
    else:
        high_res_obj_img = high_res_bgr_img  # No alpha, return just the BGR image

    return high_res_obj_img
    
    
def selective_adjust_brightness_hsv(image, brightness_adjustment, min_brightness=255):
    """
    Adjust the brightness of an image in HSV color space, with selective adjustments 
    to avoid oversaturating dark areas (like black).
    
    Parameters:
    image (numpy array): Input image in RGB format.
    brightness_adjustment (float): Brightness adjustment value.
    min_brightness (int): Minimum brightness level for selective adjustments.
    
    Returns:
    adjusted_image (numpy array): Image with adjusted brightness, preserving dark colors.
    """
    # Ensure the image has 3 channels (BGR or RGB)
    if image.shape[2] == 4:  # In case the image has an alpha channel
        image = image[:, :, :3]  # Drop the alpha channel and keep only the RGB part

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Adjust the brightness by modifying the value (V) channel
    h, s, v = cv2.split(hsv_image)
    
    # Apply selective brightness adjustment:
    # - For pixels with brightness < min_brightness, adjust less.
    # - For brighter pixels, adjust normally.
    mask = v >= min_brightness
    v[mask] = np.clip(v[mask] + brightness_adjustment, 0, 255).astype(np.uint8)
    
    # Merge channels back and convert to BGR
    adjusted_hsv = cv2.merge([h, s, v])
    adjusted_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_image


def project_shape_and_bbox(object_img, bbox_shape, field_img, real_object_width,
                           altitude, real_ground_width, offset=5):
    """
    Apply dynamic resizing, rotation, and homography projection to the shape and bounding boxes. 
    Adjust lighting to match the background while preserving the object's color.
    
    Parameters:
    object_img (numpy array): Rotated object image (RGBA).
    bbox_shape (list of tuples): Bounding box for the object.
    field_img (numpy array): The aerial image.
    real_object_width (float): Real-world width of the object in meters.
    altitude (float): Altitude of the aerial image in meters.
    real_ground_width (float): Real-world ground width captured by the aerial image in meters.
    offset (int): Maximum distortion offset for perspective projection.

    Returns:
    final_image (numpy array): Field image with the projected object.
    projected_bbox_shape (list of tuples): Projected bounding box of the shape.
    """

    # Ensure the object image has 4 channels (RGBA)
    assert object_img.shape[2] == 4, "object_img must have 4 channels (RGBA)."

    # Resize the object dynamically based on altitude and real-world dimensions
    resized_object = resize_dynamically(object_img, field_img, real_object_width, altitude, real_ground_width)
    
    # Resize the bounding box to match the resized object
    resized_object_h, resized_object_w = resized_object.shape[:2]
    bbox_shape_resized = [(x * resized_object_w / object_img.shape[1], y * resized_object_h / object_img.shape[0]) for (x, y) in bbox_shape]

    # Rotate the resized shape and its bounding box
    rotated_object, rotated_bbox_shape = rotate_shape_and_bbox(resized_object, bbox_shape_resized)
    print("resized_object shape", resized_object.shape)
    print("rotated_object shape", rotated_object.shape)
    
    # Get shape and field dimensions after rotation
    shape_rows, shape_cols = rotated_object.shape[:2]
    field_height, field_width = field_img.shape[:2]
    
    # Random top-left position for the projection (within field boundaries)
    top_left_x = random.randint(0, field_width - shape_cols)
    top_left_y = random.randint(0, field_height - shape_rows)
    
    # Define points for the homography transformation of the shape
    shape_pts = np.float32([[0, 0], [shape_cols, 0], [shape_cols, shape_rows], [0, shape_rows]])

    # Clamp offset to avoid negative values
    offset = max(0, offset)

    # Define random field points with small distortion for a mild perspective effect
    field_pts = np.float32([
        [top_left_x + random.uniform(-offset, offset), top_left_y + random.uniform(-offset, offset)],
        [top_left_x + shape_cols + random.uniform(-offset, offset), top_left_y + random.uniform(-offset, offset)],
        [top_left_x + shape_cols + random.uniform(-offset, offset), top_left_y + shape_rows + random.uniform(-offset, offset)],
        [top_left_x + random.uniform(-offset, offset), top_left_y + shape_rows + random.uniform(-offset, offset)]
    ])
    

    # Compute homography matrix
    M_homography = cv2.getPerspectiveTransform(shape_pts, field_pts)

    # Apply perspective transformation to the shape image
    transformed_shape = cv2.warpPerspective(rotated_object, M_homography, (field_img.shape[1], field_img.shape[0]))

    # Extract alpha channel for blending
    mask = transformed_shape[:, :, 3]  # Extract alpha channel
    
    mask_inv = cv2.bitwise_not(mask)   # Inverse mask for background
    transformed_shape_bgr = transformed_shape[:, :, :3]  # RGB part of the transformed shape

    # Prepare the field background using the inverse mask
    field_bg = cv2.bitwise_and(field_img, field_img, mask=mask_inv)

    # Prepare the transformed shape foreground using the alpha mask
    shape_fg = cv2.bitwise_and(transformed_shape_bgr, transformed_shape_bgr, mask=mask)

    # -------- Adjust lighting of the shape to match the background -------- #

    # Crop the region of the field where the object is placed
    field_region = field_img[top_left_y:top_left_y+shape_rows, top_left_x:top_left_x+shape_cols]
    
    # Compute average brightness in the cropped area of the field and the object
    avg_field_brightness = np.mean(cv2.cvtColor(field_region, cv2.COLOR_BGR2HSV)[:, :, 2])  # Value (V) channel
    avg_shape_brightness = np.mean(cv2.cvtColor(shape_fg, cv2.COLOR_BGR2HSV)[:, :, 2])

    # Compute brightness adjustment factor
    brightness_adjustment = avg_field_brightness - avg_shape_brightness

    # Adjust brightness of the object in HSV color space, with selective masking for dark areas
    shape_fg_adjusted = selective_adjust_brightness_hsv(shape_fg, brightness_adjustment)

    # -------- Perform blending using alpha channel -------- #

    alpha = mask.astype(float) / 255.0
    final_image = np.zeros_like(field_img)
    for c in range(3):  # Apply blending for each color channel
        final_image[:, :, c] = (alpha * shape_fg_adjusted[:, :, c] + (1 - alpha) * field_bg[:, :, c])

    # Apply homography transformation to the bounding box points
    projected_bbox_shape = cv2.perspectiveTransform(np.float32([rotated_bbox_shape]).reshape(-1, 1, 2), M_homography).reshape(-1, 2)

    # Draw the bounding box on the final image
    projected_bbox_shape = np.int32(projected_bbox_shape)  # Convert to integer for drawing
    cv2.polylines(final_image, [projected_bbox_shape], isClosed=True, color=(0, 255, 0), thickness=2)  # Green bounding box

    return final_image, projected_bbox_shape


