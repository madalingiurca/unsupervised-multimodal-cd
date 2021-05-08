import numpy as np
import sklearn
import torch
from matplotlib import pyplot as plt
from patchify import unpatchify
from skimage.filters import threshold_otsu
from tqdm import tqdm

from src.neuralNetwork.ace_net import AceNet
from src.utils.CaliforniaFloodDataModule import CaliforniaFloodDataModule

if __name__ == '__main__':

    model = AceNet()
    model = model.load_from_checkpoint(
        r"C:\Users\mgiur\PycharmProjects\UnsupervisedMultimodalCD\checkpoints\epoch=14-step=1319.ckpt")
    print(model)
    model.eval()

    test_patch_size = 250

    datamodule = CaliforniaFloodDataModule(patch_size=test_patch_size, window_step=test_patch_size, batch_size=1)
    mask = datamodule.setup()["roi"]
    dataloader = datamodule.test_dataloader()
    # diff_image = np.zeros((3500, 2000))
    patches = []

    for step, item in enumerate(tqdm(dataloader)):
        x, y, ground_truth = item
        if step % 30 == 0:
            plt.figure(), plt.imshow(y[0]), plt.show()
            plt.figure(), plt.imshow((x[0, :, :, [3, 2, 1]] + 1) / (2 * x.max()))

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        x_hat, y_hat = model(x, y)

        x_hat = x_hat.permute(0, 2, 3, 1).detach().numpy()[0]
        y_hat = y_hat.permute(0, 2, 3, 1).detach().numpy()[0]

        if step % 30 == 0:
            plt.figure(), plt.imshow(y_hat), plt.show()
            plt.figure(), plt.imshow((x_hat[:, :, [3, 2, 1]] + 1) / (2 * x.max()))

        x = x.permute(0, 2, 3, 1)[0]
        y = y.permute(0, 2, 3, 1)[0]
        # out_image_t1 = np.sum((x - x_hat) ** 2, 0)

        # Compute 2nd norm of image diff
        out_image_t1 = torch.linalg.norm((x - x_hat), ord=2, axis=-1)
        # Clip the value out of range 3*std to 3*std
        out_image_t1[out_image_t1 > torch.mean(out_image_t1) + 3 * torch.std(out_image_t1)] = torch.mean(
            out_image_t1) + 3 * torch.std(out_image_t1)
        # Normalize
        out_image_t1 = out_image_t1 / out_image_t1.max()

        out_image_t2 = torch.linalg.norm((y - y_hat), ord=2, axis=-1)
        # out_image_t2 = torch.sum((y[0] - y_hat[0]) ** 2, 0).detach().numpy()
        out_image_t2[out_image_t2 > torch.mean(out_image_t2) + 3 * torch.std(out_image_t2)] = torch.mean(
            out_image_t2) + 3 * torch.std(out_image_t2)
        out_image_t2 = out_image_t2 / out_image_t2.max()

        diff_patch = (out_image_t1 + out_image_t2) / 2.0
        if step % 30 == 0:
            plt.figure(), plt.imshow(diff_patch), plt.show()
            plt.figure(), plt.imshow(ground_truth[0]), plt.show()

        patches.append(diff_patch.detach().numpy())

    diff_image = np.array(patches)
    diff_image = unpatchify(np.array(diff_image).reshape(
        (3500 // test_patch_size, 2000 // test_patch_size, test_patch_size, test_patch_size)), (3500, 2000))
    # diff_image = np.reshape(diff_image, (3500, 2000))
    plt.figure(), plt.imshow(diff_image), plt.show()

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
    img_segm = diff_image < threshold
    plt.figure(), plt.imshow(img_segm, cmap='binary'), plt.show()

    OA = sklearn.metrics.accuracy_score(mask.flatten(), img_segm.flatten())
    AUC = sklearn.metrics.roc_auc_score(mask.flatten(), diff_image.flatten())
    AUPRC = sklearn.metrics.average_precision_score(mask.flatten(), diff_image.flatten())

    PREC_0 = sklearn.metrics.precision_score(mask.flatten(), img_segm.flatten(), pos_label=0)
    PREC_1 = sklearn.metrics.precision_score(mask.flatten(), img_segm.flatten())
    REC_0 = sklearn.metrics.recall_score(mask.flatten(), img_segm.flatten(), pos_label=0)
    REC_1 = sklearn.metrics.recall_score(mask.flatten(), img_segm.flatten())
    KC = sklearn.metrics.cohen_kappa_score(mask.flatten(), img_segm.flatten())
    [[TN, FP], [FN, TP]] = sklearn.metrics.confusion_matrix(
        mask.flatten(), img_segm.flatten()
    )
    print("MODEL METRICS:")
    print(f"Overall accuracy when otsu is used: {OA}")
    print(f"""
    area under curve: {AUC}
    precision 0, 1 = {PREC_0}, {PREC_1},
    recall 0, 1 = {REC_0}, {REC_1},
    kappa score = {KC},
    TN, FP, FN, TP = {TN}, {FP}, {FN}, {TP},
    average_precision_score = {AUPRC},
    """)
