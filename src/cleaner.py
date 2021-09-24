import numpy as np
from skimage.measure import label
from tqdm.auto import tqdm
from copy import deepcopy
from .errors import CleanerError
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
import argparse

def select_top_k_connected_areas(markup, k):
    connected_regions = label(markup)
    region_id, region_size = np.unique(connected_regions, return_counts=True)
    regions_order = (np.argsort(region_size[1:]) + 1)[::-1] # ordering without zero
    return np.isin(connected_regions, regions_order[:k])


roi_to_slices = lambda roi: tuple([slice(*i) for i in roi])

class DoubleStepCleaner:
    def __init__(self, bounding_boxes=1, connected_areas_per_label=1, minimal_box_side=50, bbox_cleaning_type='2d'):
        """Creates the cleaner instance for the segmentation data. Cleaning is a two step process here.
        First step is cropping bounding box around the main target. Second step is removing small redundant segments inside the bounding box. 
        
        Cropping is performed under assumption, that the main segmentation objects form several intact groups. 
        Therefore the *bounding_boxes* parameter defines the count of the groups.
        The whole image is divided into bounding boxes in a way that labels inside this bounding box are not connected to the labels outside of it.
        Then *bounding_boxes* largest bounding boxes selected and minimal bounding box including all of them is cropped out of the image.

        Further cleaning of small redundant segments is simply guided by the

        Args:
            bounding_boxes (int, optional): [description]. Defaults to 1.
            connected_areas_per_label (int, optional): [description]. Defaults to 1.
            minimal_box_side (int, optional): [description]. Defaults to 50.
            bbox_cleaning_type (str, optional): [description]. Defaults to '2d'.
        """
        self.bb = bounding_boxes
        self.mbs = minimal_box_side
        self.bbt = bbox_cleaning_type

        if isinstance(connected_areas_per_label, (list, ListConfig)):
            self.ca = {i['id']: i['connected_regions'] for i in connected_areas_per_label}
        else:
            self.ca = connected_areas_per_label
    
    def _get_bbox_2D(self, markup):
        msk_bin = markup > 0
        axes_low_thresholds = {0: [], 1: [], 2: []}
        axes_high_thresholds = {0: [], 1: [], 2: []}
        for axis in (0, 1, 2):
            plate = select_top_k_connected_areas(msk_bin.sum(axis)>0, self.bb)
            actual_axes = list(range(3))
            actual_axes.pop(axis)
            for sub_axis in (0, 1):
                in_box = np.where(plate.sum(1-sub_axis)>0)[0]
                axes_low_thresholds[actual_axes[sub_axis]].append(in_box[0])
                axes_high_thresholds[actual_axes[sub_axis]].append(in_box[-1])

        axes_low_thresholds = {k: min(v) for k,v in axes_low_thresholds.items()}
        axes_high_thresholds = {k: max(v) for k,v in axes_high_thresholds.items()}

        bbox = [(axes_low_thresholds[i], axes_high_thresholds[i]) for i in [0, 1, 2]]

        return tuple(bbox)
    
    def _get_bbox_1D(self, markup):
        if self.bb is None:
            return (0, markup.shape[0]-1), (0, markup.shape[1]-1), (0, markup.shape[2]-1)
        
        ax_0 = markup.sum((1, 2)) > 0
        ax_1 = markup.sum((0, 2)) > 0
        ax_2 = markup.sum((0, 1)) > 0

        axes = [ax_0, ax_1, ax_2]

        if self.bb > 0:
            axes = [select_top_k_connected_areas(ax, self.bb) for ax in axes]
        
        axes = [np.where(ax)[0] for ax in axes]
        
        if min([len(ax) for ax in axes]) == 0:
            raise CleanerError('bounding box cleaner', 'bounding box is empty')
        
        return [(ax[0], ax[-1]) for ax in axes]

    def _bounding_box_clean(self, markup):
        if self.bbt == '1d':
            bbox = self._get_bbox_1D(markup)
        elif self.bbt == '2d':
            bbox = self._get_bbox_2D(markup)
        else:
            raise Exception(f'Unknown type of the cleaning bbox setup: {self.bbt}')
        for i, (f,t) in enumerate(bbox):
            if (t - f) < 50:
                raise CleanerError('bounding box cleaner', 'bounding box is too small', axis=i)

        return bbox

    def _iterative_markup_clean(self, markup):
        if self.ca is None:
            return markup

        area_iterator = None
        if isinstance(self.ca, (dict, DictConfig)):
            area_iterator = self.ca
        else:
            area_labels = list(np.unique(markup)[0:])
            if isinstance(self.ca, (list, ListConfig)):
                area_iterator = {k: v for k,v in zip(area_labels, self.ca)}
            if isinstance(self.ca, int) or isinstance(self.ca, float):
                area_iterator = {k: self.ca for k in area_labels}

        if area_iterator is None:
            raise Exception(f'connected_areas_per_label is of unexpected type: {type(self.ca)}')

        for area_id, leave_regions in area_iterator.items():
            submarkup = (markup == area_id)
            raw_vol = submarkup.sum()
            if raw_vol < 1000:
                markup[markup == area_id] = 0
                continue
                # raise CleanerError('outlier cleaner', 'raw volume too small', area_id=area_id, raw_vol=raw_vol)
            connected_regions = label(submarkup)
            region_id, region_size = np.unique(connected_regions, return_counts=True)
            regions_order = (np.argsort(region_size[1:]) + 1)[::-1] # ordering without zero

            if isinstance(leave_regions, float):
                adds_part = region_size[regions_order] / np.cumsum(region_size[regions_order])
                leave_regions = np.where(adds_part < leave_regions)[0][0]

            clean_vol = region_size[regions_order[:leave_regions+1]].sum()
            if clean_vol < 1000:
                markup[markup == area_id] = 0
                continue
                # raise CleanerError('outlier cleaner', 'processed volume too small', area_id=area_id, clean_vol=clean_vol)

            markup[np.isin(connected_regions, regions_order[leave_regions:])] = 0

        return markup

    def __call__(self, markup):
        roi = self._bounding_box_clean(markup)
        clean_markup = self._iterative_markup_clean(deepcopy(markup[roi_to_slices(roi)]))
        roi = self._bounding_box_clean(markup)

        return clean_markup, roi_to_slices(roi)
