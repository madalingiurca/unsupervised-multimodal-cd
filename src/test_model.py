import numpy as np
import pydensecrf.densecrf as dcrf
import torch
from matplotlib import pyplot as plt
from patchify import unpatchify
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian
from scipy.io import loadmat
from skimage.filters import threshold_otsu
from sklearn import metrics
from tqdm import tqdm

from src.neuralNetwork.ace_net import AceNet
from src.neuralNetwork.x_net import XNet
from src.utils.CaliforniaFloodDataModule import CaliforniaFloodDataModule


def filtering(t1, t2, d):
    print("Filtering!")
    d = d[..., np.newaxis]
    d = np.concatenate((d, 1.0 - d), axis=2)
    W = np.size(d, 0)
    H = np.size(d, 1)
    stack = np.concatenate((t1, t2), axis=2)
    CD = dcrf.DenseCRF2D(W, H, 2)
    d[d == 0] = 10e-20
    U = -(np.log(d))
    U = U.transpose(2, 0, 1).reshape((2, -1))
    U = U.copy(order="C")
    CD.setUnaryEnergy(U.astype(np.float32))
    pairwise_energy_gaussian = create_pairwise_gaussian((10, 10), (W, H))
    CD.addPairwiseEnergy(pairwise_energy_gaussian, compat=1)
    pairwise_energy_bilateral = create_pairwise_bilateral(
        sdims=(10, 10), schan=(0.1,), img=stack, chdim=2
    )
    CD.addPairwiseEnergy(pairwise_energy_bilateral, compat=1)
    Q = CD.inference(3)
    heatmap = np.array(Q)
    heatmap = np.reshape(heatmap[0, ...], (W, H))
    return heatmap


if __name__ == '__main__':

    model1 = AceNet()
    model2 = XNet()
    model = model1.load_from_checkpoint(r"checkpoints/epoch=63-step=5631.ckpt")
    print(model)
    model.eval()

    test_patch_size = 250

    datamodule = CaliforniaFloodDataModule(patch_size=test_patch_size, window_step=test_patch_size, batch_size=1)
    datamodule.setup()
    dataloader = datamodule.test_dataloader()
    # diff_image = np.zeros((3500, 2000))
    patches = []
    mat_file = loadmat(r"resources/Flood_UiT_HCD_California_2017_Luppino.mat")
    t1 = mat_file['t1_L8_clipped']
    t2 = mat_file['logt2_clipped']
    mask = mat_file['ROI']

    for step, item in enumerate(tqdm(dataloader)):
        x, y, ground_truth = item

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        x_hat, y_hat = model(x, y)

        x_hat = x_hat.permute(0, 2, 3, 1).detach().numpy()[0]
        y_hat = y_hat.permute(0, 2, 3, 1).detach().numpy()[0]

        x = x.permute(0, 2, 3, 1)[0]
        y = y.permute(0, 2, 3, 1)[0]

        out_image_t1 = torch.sum((x - x_hat) ** 2, -1)
        # Compute 2nd norm of image diff
        # out_image_t1 = torch.linalg.norm((x - x_hat), ord=2, axis=-1)
        # Normalize
        out_image_t1 = out_image_t1 / out_image_t1.max()

        out_image_t2 = torch.sum((y - y_hat) ** 2, -1)
        # Compute 2nd norm of image diff
        # out_image_t2 = torch.linalg.norm((y - y_hat), ord=2, axis=-1)
        # Normalize
        out_image_t2 = out_image_t2 / out_image_t2.max()

        diff_patch = (out_image_t1 + out_image_t2) / 2.0
        if step % 20 == 0:
            fig, axs = plt.subplots(3, 2)
            axs[0, 0].set_title("t1")
            axs[0, 0].imshow((x[:, :, [3, 2, 1]] + 1) / (2 * x.max()))
            axs[0, 1].set_title("t2")
            axs[0, 1].imshow(y)
            axs[1, 0].set_title("x_hat")
            axs[1, 0].imshow((x_hat[:, :, [3, 2, 1]] + 1) / (2 * x.max()))
            axs[1, 1].set_title("y_hat")
            axs[1, 1].imshow(y_hat)
            axs[2, 0].set_title("difference image")
            axs[2, 0].imshow(diff_patch)
            axs[2, 1].set_title("ground truth")
            axs[2, 1].imshow(ground_truth[0])
            # plt.figure(), plt.imshow(y[0]), plt.show()
            # plt.figure(), plt.imshow((x[0, :, :, [3, 2, 1]] + 1) / (2 * x.max()))
            # plt.figure(), plt.imshow(y_hat), plt.show()
            # plt.figure(), plt.imshow((x_hat[:, :, [3, 2, 1]] + 1) / (2 * x.max()))
            # plt.figure(), plt.imshow(diff_patch), plt.show()
            # plt.figure(), plt.imshow(ground_truth[0]), plt.show()
            plt.show()
        patches.append(diff_patch.detach().numpy())

    diff_image = np.array(patches)
    diff_image = unpatchify(np.array(diff_image).reshape(
        (3500 // test_patch_size, 2000 // test_patch_size, test_patch_size, test_patch_size)), (3500, 2000))
    # diff_image = np.reshape(diff_image, (3500, 2000))
    # heatmap = filtering(t1, t2, diff_image)
    plt.figure(), plt.imshow(diff_image), plt.colorbar(), plt.show()
    diff_image[diff_image > np.mean(diff_image) + 3 * np.std(diff_image)] = np.mean(diff_image) + 3 * np.std(diff_image)
    plt.figure(), plt.imshow(diff_image), plt.colorbar(), plt.show()
    # plt.figure(), plt.imshow(heatmap), plt.show()

    # np.save('diff_image.npy', diff_image)

    # k-means clustering
    # k_m = KMeans(n_clusters=2)
    # k_m.fit(diff_image.reshape((-1, 1)))
    # Get the coordinates of the clusters centres as a 1D array
    # values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    # labels = k_m.labels_
    # img_segm = np.choose(labels, values)
    # img_segm.shape = diff_image.shape
    # img_segm = img_segm < np.mean(img_segm)
    # plt.imshow(img_segm, cmap='binary'), plt.show()
    # OA = sklearn.metrics.accuracy_score(mask.flatten(), img_segm.flatten())
    # print(f"Overall accuracy when k-means is used: {OA}")

    threshold = threshold_otsu(diff_image)
    img_segm = diff_image > threshold
    plt.figure(), plt.imshow(img_segm, cmap='binary'), plt.show()
    print("MODEL METRICS:")
    OA = metrics.accuracy_score(mask.flatten(), img_segm.flatten())
    print(f"Overall accuracy when otsu is used: {OA}")
    AUC = metrics.roc_auc_score(mask.flatten(), diff_image.flatten())
    AUPRC = metrics.average_precision_score(mask.flatten(), diff_image.flatten())

    PREC_0 = metrics.precision_score(mask.flatten(), img_segm.flatten(), pos_label=0)
    PREC_1 = metrics.precision_score(mask.flatten(), img_segm.flatten())
    REC_0 = metrics.recall_score(mask.flatten(), img_segm.flatten(), pos_label=0)
    REC_1 = metrics.recall_score(mask.flatten(), img_segm.flatten())
    KC = metrics.cohen_kappa_score(mask.flatten(), img_segm.flatten())
    [[TN, FP], [FN, TP]] = metrics.confusion_matrix(
        mask.flatten(), img_segm.flatten()
    )
    print(f"""
    area under curve: {AUC}
    precision 0, 1 = {PREC_0}, {PREC_1},
    recall 0, 1 = {REC_0}, {REC_1},
    kappa score = {KC},
    TN, FP, FN, TP = {TN}, {FP}, {FN}, {TP},
    average_precision_score = {AUPRC},
    """)
