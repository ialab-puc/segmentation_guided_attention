from PIL import Image as PILImage
import numpy as np
from torch import nn

def segmentation_to_image(segmentation, palette, output_size=(244, 244)):
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    print(segmentation.unsqueeze(0).size())
    segmentation = interp(segmentation.unsqueeze(0)).cpu().numpy().transpose(0,2,3,1)
    print(segmentation[0,1,2,:])
    seg_pred = np.asarray(np.argmax(segmentation, axis=3), dtype=np.uint8)
    print(seg_pred[0])
    output_im = PILImage.fromarray(seg_pred[0])
    output_im.putpalette(palette)
    return output_im

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