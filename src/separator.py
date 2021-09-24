from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture
import numpy as np
from skimage.measure import label
from src.errors import SeparationError


def get_centers_probabilistic(markup, count):
    gmm = GaussianMixture(n_components=count)
    gmm.fit(np.stack(np.where(markup), 1))

    return gmm.means_

def get_centers_statistical(markup, count):
    connected_regions = label(markup)
    region_id, region_size = np.unique(connected_regions, return_counts=True)
    regions_order = (np.argsort(region_size[1:]) + 1)[::-1] # ordering without zero

    centers = []
    for i in regions_order[:count]:
        centers.append(np.stack(np.where(connected_regions == i), 1).mean(0))

    return centers


class Separator:
    def __init__(self, markup, function, count=2):
        if function == 'proba':
            cf = get_centers_probabilistic
        elif function == 'stat':
            cf = get_centers_statistical
        else:
            raise SeparationError('Separator', 'Unknown type of the centering function', function=function)
        try:
            self.centers = cf(markup, count)
        except Exception as e:
            raise SeparationError('Separator', 'Failed separation function', meta=str(e))
        self.tree = KDTree(self.centers)

    def __len__(self):
        return len(self.centers)

    def __call__(self, markup):
        sm = np.zeros_like(markup, dtype=np.uint8)
        sm[markup] = self.tree.query(np.stack(np.where(markup), 1))[1] + 1
        return sm


