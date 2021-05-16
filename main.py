import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
from patchify import unpatchify
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.io import loadmat
from skimage.filters import threshold_otsu
from sklearn import metrics
from tqdm import tqdm

from src.neuralNetwork.ace_net import AceNet
from src.neuralNetwork.x_net import XNet
from src.utils.CaliforniaFloodDataModule import CaliforniaFloodDataModule

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", default="ace-net", type=str, choices=['ace-net', 'x-net'])
    arg_parser.add_argument("--verbose", help="increase verbosity", action="store_true")
    arg_parser.add_argument("--stage", default='test', type=str)
    arg_parser.add_argument("--patch_size", default=100, type=int)
    arg_parser.add_argument("--batch_size", default=8, type=int)
    arg_parser.add_argument("--epochs", default=10, type=int)
    arg_parser.add_argument("-c", "--checkpoint", type=str)

    args = arg_parser.parse_args()

    model = None

    if args.model == "ace-net":
        model = AceNet()
    elif args.model == "x-net":
        model = XNet()

    if args.stage == "train":
        datamodule = CaliforniaFloodDataModule(patch_size=args.patch_size, window_step=args.patch_size,
                                               batch_size=args.batch_size)

        trainer = Trainer(gpus=1, max_epochs=10,
                          logger=TensorBoardLogger(save_dir='lightning_logs/ACE-NET', default_hp_metric=False),
                          # auto_lr_find=True,
                          # auto_scale_batch_size=True,
                          log_every_n_steps=10,
                          flush_logs_every_n_steps=50
                          )

        trainer.fit(model, datamodule=datamodule)

    if args.stage == "test":
        if args.checkpoint is None:
            raise Exception("No model checkpoint provided.\n error: Argument -c/--checkpoint: expected one argument")

        model = model.load_from_checkpoint(args.checkpoint)
        model.eval()
        datamodule = CaliforniaFloodDataModule(patch_size=args.patch_size, window_step=args.patch_size, batch_size=1)
        datamodule.setup()
        dataloader = datamodule.test_dataloader()
        mat_file = loadmat(r"resources/Flood_UiT_HCD_California_2017_Luppino.mat")
        ground_truth = mat_file['ROI']

        patches = list()
        for step, item in enumerate(tqdm(dataloader)):
            x, y, ground_truth_patch = item
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)
            x_hat, y_hat = model(x, y)
            # plt.figure(), plt.imshow(y_hat[0].permute(1, 2, 0).detach().numpy()), plt.show()
            x_hat = x_hat.permute(0, 2, 3, 1).detach().numpy()[0]
            y_hat = y_hat.permute(0, 2, 3, 1).detach().numpy()[0]
            x = x.permute(0, 2, 3, 1)[0]
            y = y.permute(0, 2, 3, 1)[0]

            # Compute 2nd norm of image diff and normalize
            out_image_t1 = torch.sum((x - x_hat) ** 2, -1)
            out_image_t1 = out_image_t1 / out_image_t1.max()
            out_image_t2 = torch.sum((y - y_hat) ** 2, -1)
            out_image_t2 = out_image_t2 / out_image_t2.max()
            diff_patch = (out_image_t1 + out_image_t2) / 2.0

            if step % 20 == 0 and args.verbose:
                fig, axs = plt.subplots(3, 2, figsize=(3,3))
                axs[0, 0].set_title("t1")
                axs[0, 0].imshow((x[:, :, [3, 2, 1]] + 1) / (x.max()))
                axs[0, 1].set_title("t2")
                axs[0, 1].imshow(y)
                axs[1, 0].set_title("x_hat")
                axs[1, 0].imshow((x_hat[:, :, [3, 2, 1]] + 1) / (x_hat.max()))
                axs[1, 1].set_title("y_hat")
                axs[1, 1].imshow(y_hat)
                axs[2, 0].set_title("difference image")
                axs[2, 0].imshow(diff_patch)
                axs[2, 1].set_title("ground truth")
                axs[2, 1].imshow(ground_truth_patch[0])

                plt.show()

            patches.append(diff_patch.detach().numpy())

        diff_image = np.array(patches)
        diff_image = unpatchify(np.array(diff_image).reshape(
            (ground_truth.shape[0] // args.patch_size, ground_truth.shape[1] // args.patch_size, args.patch_size,
             args.patch_size)), ground_truth.shape)

        diff_image[diff_image > np.mean(diff_image) + 3 * np.std(diff_image)] = np.mean(diff_image) + 3 * np.std(
            diff_image)
        plt.figure(), plt.imshow(diff_image), plt.colorbar(), plt.show()

        threshold = threshold_otsu(diff_image)
        img_segm = diff_image > threshold
        plt.figure(), plt.imshow(img_segm, cmap='binary'), plt.show()
        print("MODEL METRICS:")
        OA = metrics.accuracy_score(ground_truth.flatten(), img_segm.flatten())
        print(f"Overall accuracy when otsu is used: {OA}")
        AUC = metrics.roc_auc_score(ground_truth.flatten(), diff_image.flatten())
        AUPRC = metrics.average_precision_score(ground_truth.flatten(), diff_image.flatten())

        PREC_0 = metrics.precision_score(ground_truth.flatten(), img_segm.flatten(), pos_label=0)
        PREC_1 = metrics.precision_score(ground_truth.flatten(), img_segm.flatten())
        REC_0 = metrics.recall_score(ground_truth.flatten(), img_segm.flatten(), pos_label=0)
        REC_1 = metrics.recall_score(ground_truth.flatten(), img_segm.flatten())
        KC = metrics.cohen_kappa_score(ground_truth.flatten(), img_segm.flatten())
        [[TN, FP], [FN, TP]] = metrics.confusion_matrix(
            ground_truth.flatten(), img_segm.flatten()
        )
        print(f"""
            area under curve: {AUC}
            precision 0, 1 = {PREC_0}, {PREC_1},
            recall 0, 1 = {REC_0}, {REC_1},
            kappa score = {KC},
            TN, FP, FN, TP = {TN}, {FP}, {FN}, {TP},
            average_precision_score = {AUPRC},
            """)
