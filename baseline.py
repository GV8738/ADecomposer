import torch
import warnings
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


import configuration
from models import Vae
from utils import utils
from dataset import prepare_dataset

categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
              'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
warnings.filterwarnings("ignore")

def soft_threshold(x, a=0.5):
    return torch.sgn(x)*torch.nn.functional.relu(x.abs() - a, 0)

def main(args):
    device = torch.device('cuda')

    # Getting dataset/dataloader
    img_transforms, mask_transforms = prepare_dataset.get_transforms(args.size)
    te_dataset = prepare_dataset.MVTecDataset(root=args.dataroot,
        target=args.category,
        train=False,
        transforms=img_transforms,
        mask_transforms=mask_transforms)
    te_dataloader = DataLoader(te_dataset, batch_size=1, shuffle=False)

    # Create model
    model = Vae.VanillaVAE(in_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256, 512]).to(device)
    weights = torch.load(args.root + '/weights/vae_' + args.category + '_200_cycle.pth')
    #weights = torch.load(args.root + '/VAE/capsule.pth')
    model.load_state_dict(weights['model'])
    #model.load_state_dict(weights['state_dict'])
    model.eval()

    scores = []
    pixels = []
    labels = []
    masks = []
    for (x, mask, label) in te_dataloader:
        #Initialization
        x = x.to(device)

        #Forward pass
        ŷ = model.generate(x)

        #Computing error
        ε = (x - ŷ)**2

        #Anomaly score
        A_map = ε.prod(1)
        A_score = ε.max().cpu().item()
        scores.append(A_score)
        pixels.extend(utils.min_max_norm(A_map).flatten().detach().cpu().numpy())
        utils.relabel(label)
        labels.append(label.item())
        masks.extend(mask.flatten().cpu().numpy().astype(int))

    scores = np.array(scores)
    pixels = np.array(pixels)
    labels = np.array(labels)
    masks = np.array(masks)

    #Image-level AUROC
    img_auroc = roc_auc_score(labels, scores)

    #Pixel-level AUROC
    pxl_auroc = roc_auc_score(masks, pixels)

    print('Image-level AUROC: ', img_auroc)
    print('Pixel-level AUROC: ', pxl_auroc)


if __name__ == '__main__':
    args = configuration.argparser()
    main(args)