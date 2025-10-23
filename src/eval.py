## -*- coding: utf-8 -*-
import os, sys
sys.setrecursionlimit(15000)
import torch
import numpy as np
import random
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import time
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm
import timm
from src.utils.util import *
from src.models.ViT_MoE import *

from src.datasets import dataloader
import csv

def setup_seed(seed):
    print('Using device:' , torch.cuda.is_available())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def test(config, model, test_loader, model_path):
    """
    Evaluate model on test set and save results to a CSV file.
    """
    # === Load checkpoint ===
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Start test mode...')
    model.eval()
    video_predictions, video_labels = [], []
    frame_predictions, frame_labels = [], []

    # === Create save dir ===
    os.makedirs(config.save_dir, exist_ok=True)
    csv_path = os.path.join(config.save_dir, f"test_results_{config.dataset.name}_{config.version}.csv")

    with torch.no_grad():
        st_time = time.time()

        for inputs, labels in tqdm(test_loader, total=len(test_loader), ncols=70, leave=False, unit='step'):
            inputs = inputs.cuda()
            labels = labels.cuda()

            # If inputs are [1, N, 3, H, W] (video tensor), flatten into frames
            if inputs.ndim == 5:
                inputs = inputs.squeeze(0)       # [N, 3, H, W]
            if inputs.ndim == 3:
                inputs = inputs.unsqueeze(0)     # ensure batch dim
            inputs = inputs.view(-1, 3, inputs.shape[-2], inputs.shape[-1])

            # --- Forward ---
            outputs, _ = model(inputs)
            outputs = F.softmax(outputs, dim=-1)

            # --- Frame-level metrics ---
            frame_count = outputs.shape[0]
            frame_predictions.extend(outputs[:, 1].cpu().tolist())     # probability of fake
            frame_labels.extend(labels.expand(frame_count).cpu().tolist())

            # --- Video-level prediction ---
            video_score = torch.mean(outputs[:, 1])                    # mean prob(fake)
            video_predictions.append(video_score.cpu().item())
            video_labels.append(labels.cpu().item())

        period = time.time() - st_time
    print(f'Total inference time: {period:.2f}s')

    # === Compute metrics ===
    frame_results = cal_metrics(frame_labels, frame_predictions, threshold=0.5)
    video_results = cal_metrics(video_labels, video_predictions, threshold=0.5)

    print('Test result:')
    print(f'  Video: ACC={video_results.ACC:.2%}, AUC={video_results.AUC:.4f}, EER={video_results.EER:.2%}')
    print(f'  Frame: ACC={frame_results.ACC:.2%}, AUC={frame_results.AUC:.4f}, EER={frame_results.EER:.2%}')

    # === Save to CSV ===
    header = [
        "timestamp", "dataset", "model", "checkpoint",
        "video_acc", "video_auc", "video_eer",
        "frame_acc", "frame_auc", "frame_eer",
        "test_time(s)"
    ]

    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        getattr(config.dataset, 'name', 'unknown'),
        getattr(config.model, 'backbone', 'unknown'),
        os.path.basename(model_path),
        f"{video_results.ACC:.4f}",
        f"{video_results.AUC:.4f}",
        f"{video_results.EER:.4f}",
        f"{frame_results.ACC:.4f}",
        f"{frame_results.AUC:.4f}",
        f"{frame_results.EER:.4f}",
        f"{period:.2f}"
    ]

    # append or create file
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"✅ Results saved to {csv_path}")

if __name__ == '__main__':
    from src.configs.config_parser import load_config
    config = load_config('src/configs/eval_config.yaml')
    start_time = time.time()
    setup_seed(2024)

    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(config.device)
    print('Using device: {}'.format(config.device), torch.cuda.is_available())
    device = torch.device(config.device) if torch.cuda.is_available() else "cpu"

    test_loader = dataloader.get_dataloader(
        config.dataset.root,
        dataset_name=config.dataset.name,
        batch_size=1,
        phase="test"
    )

    model = vit_base_patch16_224_in21k(pretrained=True,num_classes=2)
    model = model.cuda()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    print('Start eval process...')

    model_path = config.model.path
    current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not model_path:
        model_path = f"{current_folder}/{config.save_dir}/models_params_best.tar"
    print(model_path)
    test(config, model,test_loader,model_path)
    duration = time.time()-start_time
    # print('The best AUC is {:.2%}'.format(auc))
    print('The run time is {}h {}m'.format(int(duration//3600),int(duration%3600//60)))
