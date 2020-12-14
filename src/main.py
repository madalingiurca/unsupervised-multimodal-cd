from src.dataPrepare import process_image
from matplotlib import pyplot as plt

if __name__ == '__main__':
    var1 = process_image()
    print(var1.shape)
    plt.plot(var1)