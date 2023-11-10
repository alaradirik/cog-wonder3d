import os
import cv2
import rembg
import numpy as np
from rembg import remove


def preprocess(image_path, remove_bg=True, img_size=256):
    border_ratio = 0.2
    recenter = True

    out_base = os.path.basename(image_path).split('.')[0]
    out_rgba = os.path.join('/src/image.png')
    
    # load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] != 4 and remove_bg == False:
        raise ValueError("Please provide an RGBA image with background removed or set remove_bg=True.")
    
    # RGBA image of shape [height, width, 4]
    if remove_bg:
        print(f'[INFO] background removal...')
        carved_image = remove(image)
    else:
        carved_image = image

    mask = carved_image[..., -1] > 0

    # Recenter image
    if recenter:
        final_rgba = np.zeros((img_size, img_size, 4), dtype=np.uint8)
        
        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        height = x_max - x_min
        width = y_max - y_min
        desired_size = int(img_size * (1 - border_ratio))
        scale = desired_size / max(height, width)

        height_new = int(height * scale)
        width_new = int(width * scale)
        x2_min = (img_size - height_new) // 2
        x2_max = x2_min + height_new
        y2_min = (img_size - width_new) // 2
        y2_max = y2_min + width_new
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            carved_image[x_min:x_max, y_min:y_max], 
            (width_new, height_new), 
            interpolation=cv2.INTER_AREA
        )
    else:
        final_rgba = carved_image
    
    # write image
    cv2.imwrite(out_rgba, final_rgba)
    return out_rgba