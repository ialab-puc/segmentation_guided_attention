from PIL import Image as PILImage
import numpy as np
from torch import nn
import cv2

def segmentation_to_image(segmentation, palette, output_size=(244, 244)):
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    segmentation = interp(segmentation.unsqueeze(0)).cpu().numpy().transpose(0,2,3,1)
    seg_pred = np.asarray(np.argmax(segmentation, axis=3), dtype=np.uint8)
    output_im = PILImage.fromarray(seg_pred[0])
    output_im.putpalette(palette)
    return output_im

def attention_to_images(image,attention_map,output_size=(244,244), normalize='local'):
    cvImage = cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR) #we neeed a 3 dimensional gray image
    cvImage = cv2.resize(cvImage, output_size)
    attention_map = attention_map[:,:,0:960]
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    attention_map = attention_map.mean(dim=2, keepdim=True)
    attention_size = attention_map.size()
    dim = int(attention_size[1]**(0.5))
    attention_map = attention_map.view((attention_size[0],1,dim,dim))
    attention_map = interp(attention_map).squeeze(1).cpu().detach().numpy()
    return global_normalize(cvImage, attention_map) if normalize == 'global' else local_normalize(cvImage, attention_map)

def global_normalize(image,attention_map):
    images = []
    heatmap_img = None
    heatmap_img = cv2.normalize(attention_map, heatmap_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    for single_map in heatmap_img:
        single_img = cv2.applyColorMap(single_map, cv2.COLORMAP_JET)
        result = cv2.addWeighted(single_img, 0.5, image, 0.5, 0)
        images.append(result)
    return images

def local_normalize(image,attention_map):
    images = []
    for single_map in attention_map:
        heatmap_img = None
        heatmap_img = cv2.normalize(single_map, heatmap_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        result = cv2.addWeighted(img, 0.5, image, 0.5, 0)
        images.append(result)
    return images

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette