from .ellipsoid_tool import EllipsoidTool
import numpy as np
from skimage.segmentation import find_boundaries
from skimage.measure import label
from .separator import get_centers_probabilistic, get_centers_statistical
from itertools import combinations
from .errors import MeasurementError

def recurrent_cleaner(s):
    if isinstance(s, list):
        return [recurrent_cleaner(i) for i in s]

    if isinstance(s, tuple):
        return tuple([recurrent_cleaner(i) for i in s])

    if isinstance(s, int):
        return s

    if isinstance(s, float):
        return s

    if isinstance(s, np.integer):
        return int(s)

    if isinstance(s, np.floating):
        return float(s)

    raise Exception(f'Unexpected type for serialisation: {type(s)} for value {s}')


def organ_measure(mf):
    def wrapped_mf(markup, volume, separator=None):
        if separator is None:
            results =  [mf(markup, volume)]
        else:
            results = []
            separated_markup = separator(markup)
            try:
                for i in range(len(separator)):
                    results.append(mf(separated_markup == i+1, volume))
            except Exception as e:
                raise MeasurementError(mf.__name__, str(e)) # no meta up to now, huh?
        return recurrent_cleaner(results)
    return wrapped_mf

def axial_apply(mf):
    def wrapped_mf(markup, volume):
        center = [i.mean() for i in np.where(markup)]
        axial_measures = []
        for axis in (0, 1, 2):
            line = [int(c) for c in center]
            line[axis] = slice(None)
            center_along_axis = center[axis]
            axial_measures.append(mf(markup, line, center_along_axis))
        return axial_measures

    return wrapped_mf


@organ_measure
def volume(markup, volume):
    """Sums up volume of the markup"""
    return markup.sum()

@organ_measure
def surface_area(markup, volume):
    """Actually, calculates volume of pixels determined as boundary pixels by skimage"""
    return find_boundaries(markup).sum()

@organ_measure
def color_average(markup, volume):
    """Calculates average value inside segmented organ"""
    return volume[markup].mean()

@organ_measure
def color_std(markup, volume):
    """Calculates variance of values inside segmented organ"""
    return volume[markup].std()

@organ_measure
@axial_apply
def thickness_axial(markup, line, center):
    """For each direction along of one of three axis counts distance between first and last segmented pixels"""
    touched = np.where(markup[tuple(line)])[0]
    t1 = touched[touched < center][-1] - touched[0]
    t2 = touched[-1] - touched[touched > center][0]
    return (t1, t2)

@organ_measure
@axial_apply
def radius_axial(markup, line, center):
    """For each direction along of one of three axis calculate distance between center and last segmented pixel"""
    touched = np.where(markup[tuple(line)])[0]
    r1 = center - touched[0]
    r2 = touched[-1] - center
    return (r1, r2)

def get_bootstrapped_radii(markup, samples=10, size=5000):
    point_cloud = np.stack(np.where(markup), 1)

    all_radii = []
    for i in range(samples):
        eltool = EllipsoidTool()
        ell = eltool.getMinVolEllipse(point_cloud[np.random.choice(len(point_cloud), size, replace=False)])
        all_radii.append(np.sort(ell[1]))
    all_radii = np.stack(all_radii, 0)

    median_radii = np.median(all_radii, 0)
    return median_radii

@organ_measure
def eccentricity_meridional(markup, volume):
    """Calculates average meridional eccentricity of the circumscribed ellipsoid of organ"""
    r1, r2, r3 = get_bootstrapped_radii(markup)
    return (r3 - r1) / (r3 + r1)

@organ_measure
def eccentricity_equatorial(markup, volume):
    """Calculates average equatorial eccentricity of the circumscribed ellipsoid of organ"""
    r1, r2, r3 = get_bootstrapped_radii(markup)
    return (r3 - r2) / (r3 + r2)

def distance_between_centers(markup, volume, separator):
    """Calculates distance between two pair organs"""
    if separator is None:
        centers = get_centers_probabilistic(markup, 2)
    else:
        separate_markup = separator(markup)
        centers = []
        for i in range(len(separator)):
            centers.append(np.array([i.mean() for i in np.where(separate_markup == i+1)]))

    d = []
    for a,b in combinations(centers, 2):
        d.append(((a-b)**2).sum()**0.5)
    return d
