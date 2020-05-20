"""Train script.

Usage:
    train.py <hparams> <dataset> <dataset_root>
"""
import os
import vision
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.trainer import Trainer
from glow.config import JsonConfig

if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]
    dataset_root = args["<dataset_root>"]
    hparams = JsonConfig(hparams)
    dataset = vision.Datasets[dataset]
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])
    # build graph and dataset
    dataset = dataset(dataset_root, transform=transform)
    # build graph and dataset
    try:
        built = build(hparams, dataset, True)
    except SystemExit as e:
        if str(e) == "123":
            print("init the parameters with first batch samples...")
            built = build(hparams, dataset, True)
        else:
            print("programm exited with unkown reason")
            system.exit(0)
    # begin to train
    trainer = Trainer(**built, dataset=dataset, dataset_root=dataset_root, hparams=hparams)
    if not hparams.Glow.reverse:
        trainer.train()
    else:
        trainer.infer()
