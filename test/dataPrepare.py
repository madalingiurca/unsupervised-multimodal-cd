from scipy.io import loadmat
from skimage.util import view_as_windows


def process_image():
    mat = loadmat('resources/Flood_UiT_HCD_California_2017_Luppino.mat')
    roi = mat['ROI']
    t1_image = mat['t1_L8_clipped']
    t2_image = mat['logt2_clipped']

    print(t2_image.shape)
    t1_image = view_as_windows(t1_image, (16, 16, 11)).squeeze()
    t2_image = view_as_windows(t2_image, (16, 16, 3)).squeeze()
    return roi, t1_image, t2_image
