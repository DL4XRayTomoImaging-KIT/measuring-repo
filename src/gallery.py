import os
import random

import numpy as np
from PIL import Image
from pathlib import Path
from tinydb import TinyDB, Query, where


# Hardcoded so far, TODO: move to config
recolor_dict = {'bg': {'label': 0, 'color':(0, 0, 0)},
                'iris': {'label': 1, 'color':(81, 7, 230)},
                'lens': {'label': 3, 'color':(107, 163, 237)},
                'muscles': {'label': 4, 'color':(219, 97, 64)},
                'nerve': {'label': 5, 'color':(245, 238, 140)},
                'retina': {'label': 6, 'color':(111, 153, 107)}
                }

views = {'front':0, 'top':1, 'side':2}

def convert_label_to_color(seg, recolor_params=None):

    #color = np.empty([3, seg.shape[0], seg.shape[1]])

    color = np.zeros((seg.shape[0],seg.shape[1],3), np.uint8)

    #if recolor_params == None:
    #    return

    for key in recolor_params:
        i = recolor_params[key]['label']
        c = recolor_params[key]['color']

        ind = np.where(seg == i)
        color[ind] = c

    return color




def make_gallery(volume, segmentation, params):

    keep_every_slice = params['keep_every_slice']
    sample_id = params['sample_id']
    gallery_path = Path(params['gallery_path'])
    #slices_range = params['slices_range']
    scale_small = params['scale_small']
    scale_large= params['scale_large']
    blend_alpha= params['blend_alpha']

    if not os.path.exists(gallery_path):
        os.makedirs(gallery_path)

    print(f'Making gallery for sample {sample_id}')
    print(volume.shape)
    print(segmentation.shape)

    #db.search(where('id') == '101')

#id[0]['group_bbox']

    # x1 = slices_range[2][0]
    # x2 = slices_range[2][1]
    # y1 = slices_range[1][0]
    # y2 = slices_range[1][1]
    # z1 = slices_range[0][0]
    # z2 = slices_range[0][1]

    for view in views.keys():

        for i in range(0, volume.shape[views[view]], keep_every_slice):

            if view == 'front':
                slice_vol = volume[i,:,:]
                slice_seg = segmentation[i,:,:]
            elif view == 'top':
                slice_vol = volume[:,i,:]
                slice_seg = segmentation[:,i,:]
            else:
                slice_vol = np.flipud(np.rot90(volume[:,:,i]))
                slice_seg = np.flipud(np.rot90(segmentation[:,:,i]))

            im_colored_label= Image.fromarray(convert_label_to_color(slice_seg, recolor_dict), mode='RGB')
            im_overlay = Image.blend(Image.fromarray(slice_vol).convert('RGB'), im_colored_label, alpha=blend_alpha)
    
            if not os.path.exists(gallery_path / sample_id / view):
                os.makedirs(gallery_path / sample_id / view)

            if not os.path.exists(gallery_path / sample_id / view / 'large'):
                os.makedirs(gallery_path / sample_id / view / 'large')

            im_resized = im_overlay.resize((int(im_overlay.width * scale_small), int(im_overlay.height*scale_small)))
            im_resized.save(gallery_path / sample_id / view / f'overlay_{str(i).zfill(4)}.jpg', 'JPEG')

            if scale_large == 1.0:
                im_resized = im_overlay
            else:
                im_resized = im_overlay.resize((int(im_overlay.width * scale_large), int(im_overlay.height*scale_large)))
                im_resized.save(gallery_path / sample_id / view / 'large' / f'overlay_{str(i).zfill(4)}.jpg', 'JPEG')

                

    #return (int(volume.shape[0] / keep_every_slice), int(volume.shape[1] / keep_every_slice), int(volume.shape[2] / every_slice))
