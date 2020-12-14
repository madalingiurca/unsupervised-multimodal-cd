from scipy.io import loadmat


def process_image():
    mat = loadmat('resources/Flood_UiT_HCD_California_2017_Luppino.mat')
    return mat
