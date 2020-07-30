from PIL import Image as PILImage
import numpy as np
from torch import nn
import cv2

def segmentation_to_image(segmentation,image,palette, output_size=(244, 244)):
    grayed = gray_image(image,output_size)
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    segmentation = interp(segmentation.unsqueeze(0)).cpu().numpy().transpose(0,2,3,1)
    seg_pred = np.asarray(np.argmax(segmentation, axis=3), dtype=np.uint8)
    output_im = PILImage.fromarray(seg_pred[0])
    output_im.putpalette(palette)
    output_im = output_im.convert('RGB')
    result = cv2.addWeighted(np.array(output_im), 0.7, grayed, 0.3, 0)
    return result

def attention_to_images(image,attention_map,output_size=(244,244), normalize='local'):
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    cvImage = gray_image(image,output_size)
    attention_map = interp(attention_map).squeeze(1).cpu().detach().numpy()
    ticks=[(attention_map.min()),(attention_map.max())]
    normalized = global_normalize(cvImage, attention_map) if normalize == 'global' else local_normalize(cvImage, attention_map)
    return normalized, ticks

def shape_attention(attention_map):
    attention_map = attention_map.mean(dim=1, keepdim=True).permute([0,2,1])
    attention_size = attention_map.size()
    dim = int(attention_size[1]**(0.5))
    attention_map = attention_map.permute([0,2,1]).view((attention_size[0],1,dim,dim))
    return attention_map

def gray_image(image,output_size):
    cvImage = cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR) #we neeed a 3 dimensional gray image
    cvImage = cv2.resize(cvImage, output_size)
    return cvImage

def global_normalize(image,attention_map, mask_value=-1):
    images = []
    heatmap_img = None
    heatmap_img = normalize_attention(attention_map, mask_value)
    for single_map in heatmap_img:
        single_img = cv2.applyColorMap(single_map, cv2.COLORMAP_JET)
        result = cv2.addWeighted(single_img, 0.5, image, 0.5, 0)
        result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        images.append(result)
    return images

def local_normalize(image,attention_map, mask_value=-1):
    images = []
    for single_map in attention_map:
        heatmap_img = None
        heatmap_img = normalize_attention(single_map, mask_value)
        img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        result = cv2.addWeighted(img, 0.5, image, 0.5, 0)
        result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        images.append(result)
    return images

def normalize_attention(attention, mask_value=-1):
    input_min = np.min(attention[attention > mask_value])
    input_max = attention.max()
    return (np.maximum((attention - input_min),0)/(input_max - input_min) * 255).astype('uint8')

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
