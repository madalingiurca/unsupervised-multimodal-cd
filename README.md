# Unsupervised Multimodal Change Detection

Development of ACE-NET deep-neural-network based on [ArXiv paper](https://arxiv.org/pdf/2001.04271.pdf)

The purpose of this method focus on highlighting changes between satellites' captured images. 
A special neural network architecture translates the images between the two domains of different remote sensors.

Uses `Flood_UiT_HCD_California_2017_Luppino` as training database.

Test the model using `python main.py -c checkpoints/epoch=249-step=21999.ckpt --patch_size 250 --verbose`

`-c` refers to path of the file with model's pretrained parameters. Train the network on your own or use one of checkpoints provided inside this repository.

> part of Degree thesis
