from .ellipsoid_tool import EllipsoidTool
import numpy as np
from skimage.segmentation import find_boundaries
from skimage.measure import label
from .separator import get_centers_probabilistic, get_centers_statistical
from itertools import combinations
from .errors import MeasurementError
from scipy.spatial import ConvexHull, Voronoi
from scipy.spatial.distance import cdist
from einops import rearrange
from skimage.morphology import ball, binary_dilation, binary_erosion, remove_small_holes
from scipy.ndimage import binary_fill_holes
import miniball

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
            markup_ids = [1]
        else:
            markup = separator(markup)
            markup_ids = [i+1 for i in range(len(separator))]

        results = []
        try:
            for i in markup_ids:
                results.append(mf(markup == i, volume))
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
    
def organ_metric(markup, volume, metric, modifier=None):
    modifiers = ['dilation', 'erosion', 'filled']
    if modifier is not None and modifier not in modifiers:
        raise ValueError(f'invalid modifier, must be in {modifiers}')

    if modifier == 'dilation':
        dilated_markup = binary_dilation(markup, ball(radius=10, dtype=bool))
        markup = dilated_markup & ~markup
    elif modifier == 'erosion':
        markup = binary_erosion(markup, ball(radius=10, dtype=bool))
    elif modifier == 'filled':
        markup = binary_fill_holes(markup)

    if metric == 'volume':
        measurement = markup.sum()
    elif metric == 'surface_area':
        measurement = find_boundaries(markup).sum()    
    elif metric == 'mean':
        measurement = volume[markup].mean()
    elif metric == 'median':
        measurement = np.median(volume[markup])    
    elif metric == 'std':
        measurement = volume[markup].std()
    elif metric == 'perc_1':
        measurement = np.percentile(volume[markup], 1)
    elif metric == 'perc_99':
        measurement = np.percentile(volume[markup], 99)

    return measurement


@organ_measure
def volume(markup, volume):
    """Sums up volume of the markup"""
    return organ_metric(markup, volume, 'volume')

@organ_measure
def volume_filled(markup, volume):
    """Sums up volume of the markup after filling closed holes"""
    return organ_metric(markup, volume, 'volume', 'filled')

@organ_measure
def surface_area(markup, volume):
    """Actually, calculates volume of pixels determined as boundary pixels by skimage"""
    return organ_metric(markup, volume, 'surface_area')

@organ_measure
def color_mean(markup, volume):
    """Calculates mean value inside segmented organ"""
    return organ_metric(markup, volume, 'mean')

@organ_measure
def color_mean_dilated(markup, volume):
    """mean after dilation by 10 pixels"""
    return organ_metric(markup, volume, 'mean', 'dilation')

@organ_measure
def color_mean_eroded(markup, volume):
    """mean after erosion by 10 pixels"""
    return organ_metric(markup, volume, 'mean', 'erosion')

@organ_measure
def color_median(markup, volume):
    """Calculates median of values inside segmented organ"""
    return organ_metric(markup, volume, 'median')

@organ_measure
def color_median_dilated(markup, volume):
    """median after dilation by 10 pixels"""
    return organ_metric(markup, volume, 'median', 'dilation')

@organ_measure
def color_median_eroded(markup, volume):
    """median after erosion by 10 pixels"""
    return organ_metric(markup, volume, 'median', 'erosion')

@organ_measure
def color_std(markup, volume):
    """Calculates standard deviation of values inside segmented organ"""
    return organ_metric(markup, volume, 'std')

@organ_measure
def color_std_dilated(markup, volume):
    """std dev. after dilation by 10 pixels"""
    return organ_metric(markup, volume, 'std', 'dilation')

@organ_measure
def color_std_eroded(markup, volume):
    """std dev. after erosion by 10 pixels"""
    return organ_metric(markup, volume, 'std', 'erosion')

@organ_measure
def color_perc_99(markup, volume):
    """Calculates 99th percentile of values inside segmented organ"""
    return organ_metric(markup, volume, 'perc_99')

@organ_measure
def color_perc_99_dilated(markup, volume):
    """99th percentile after dilation by 10 pixels"""
    return organ_metric(markup, volume, 'perc_99', 'dilation')

@organ_measure
def color_perc_99_eroded(markup, volume):
    """99th percentile after erosion by 10 pixels"""
    return organ_metric(markup, volume, 'perc_99', 'erosion')

@organ_measure
def color_perc_1(markup, volume):
    """Calculates 1st percentile of values inside segmented organ"""
    return organ_metric(markup, volume, 'perc_1')

@organ_measure
def color_perc_1_dilated(markup, volume):
    """1st percentile after dilation by 10 pixels"""
    return organ_metric(markup, volume, 'perc_1', 'dilation')

@organ_measure
def color_perc_1_eroded(markup, volume):
    """1st percentile after erosion by 10 pixels"""
    return organ_metric(markup, volume, 'perc_1', 'erosion')

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

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6
    
@organ_measure
def convex_volume(markup, volume):
    """Calculates volume of the convex hull enclosing the segmented organ"""
    mp = np.dstack(np.where(markup))[0] # marked points
    ch = ConvexHull(mp)

    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex), ch.simplices))
    tets = ch.points[simplices]
    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]))

@organ_measure
def radius_minimal_sphere(markup, volume):
    """Calculates radius of the minimal sphere which contains the convex hull surrounding the label"""
    mp = np.dstack(np.where(markup))[0] # marked points
    ch = ConvexHull(mp)
    hp = ch.points[ch.vertices]  # hull vertice coordinates

    C,r2 = miniball.get_bounding_ball(hp)
    return r2**0.5

@organ_measure
def radius_maximal_sphere(markup, volume):
  """Calculates radius of the maximal sphere enclosed within the convex hull surrounding the label"""
  mp = np.dstack(np.where(markup))[0]
  ch = ConvexHull(mp)
  hp = ch.points[ch.vertices]  # hull vertice coordinates
  vor = Voronoi(hp)
  vertices = vor.vertices

  # find vertex with the largest clearance radius (distance to its defining points)
  radius = 0
  for vertex in vertices:
    distances = cdist(hp, np.expand_dims(vertex,0))
    min_distance = np.min(distances)
    if min_distance > radius:
      radius,center = min_distance, vertex
  return radius

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
