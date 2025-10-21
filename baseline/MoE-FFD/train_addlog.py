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
from dataset import FFPP_Dataset,TestDataset
import timm
from utils import *
from ViT_MoE import *
from torchvision import datasets, transforms

from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


class TrainingLogger:
    """Log training metrics to CSV and plot after each epoch"""
    
    def __init__(self, save_dir: str):
        """
        Args:
            save_dir: Directory to save CSV and plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file path
        self.csv_file = self.save_dir / 'training_metrics.csv'
        self.plot_dir = self.save_dir / 'plots'
        self.plot_dir.mkdir(exist_ok=True)
        
        # Initialize CSV file with headers
        self.headers = [
            'epoch', 'train_loss', 'train_acc',
            'video_acc', 'video_auc', 'video_eer',
            'frame_acc', 'frame_auc', 'frame_eer'
        ]
        
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
        
        # Store metrics for plotting
        self.metrics_history = {h: [] for h in self.headers}
    
    def log_epoch(self, epoch: int, metrics_dict: Dict) -> None:
        """
        Log metrics for one epoch
        
        Args:
            epoch: Epoch number
            metrics_dict: Dictionary with keys like 'train_loss', 'train_acc', 
                         'video_acc', 'video_auc', 'video_eer', 
                         'frame_acc', 'frame_auc', 'frame_eer'
        """
        # Prepare row data
        row = {'epoch': epoch}
        row.update(metrics_dict)
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row)
        
        # Update history
        for key, value in row.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Plot after each epoch
        self._plot_metrics(epoch)
    
    def _plot_metrics(self, epoch: int) -> None:
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Metrics - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        epochs_list = self.metrics_history['epoch']
        
        # Plot 1: Train Loss
        ax = axes[0, 0]
        ax.plot(epochs_list, self.metrics_history['train_loss'], 'b-o', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Train Accuracy
        ax = axes[0, 1]
        ax.plot(epochs_list, self.metrics_history['train_acc'], 'g-o', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Plot 3: Video AUC
        ax = axes[0, 2]
        ax.plot(epochs_list, self.metrics_history['video_auc'], 'r-o', linewidth=2, label='Video AUC')
        ax.plot(epochs_list, self.metrics_history['frame_auc'], 'orange', marker='s', linewidth=2, label='Frame AUC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('Video vs Frame AUC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Plot 4: Video Accuracy
        ax = axes[1, 0]
        ax.plot(epochs_list, self.metrics_history['video_acc'], 'purple', marker='o', linewidth=2, label='Video Acc')
        ax.plot(epochs_list, self.metrics_history['frame_acc'], 'brown', marker='s', linewidth=2, label='Frame Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Video vs Frame Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Plot 5: Video EER
        ax = axes[1, 1]
        ax.plot(epochs_list, self.metrics_history['video_eer'], 'cyan', marker='o', linewidth=2, label='Video EER')
        ax.plot(epochs_list, self.metrics_history['frame_eer'], 'magenta', marker='s', linewidth=2, label='Frame EER')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('EER (lower is better)')
        ax.set_title('Video vs Frame EER')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Summary Stats
        ax = axes[1, 2]
        if len(epochs_list) > 0:
            # Find best epoch by video AUC
            best_idx = np.argmax(self.metrics_history['video_auc'])
            best_epoch = epochs_list[best_idx]
            best_auc = self.metrics_history['video_auc'][best_idx]
            
            stats_text = (
                f"Best Epoch: {best_epoch}\n"
                f"Best Video AUC: {best_auc:.4f}\n"
                f"Latest Video AUC: {self.metrics_history['video_auc'][-1]:.4f}\n"
                f"Latest Video Acc: {self.metrics_history['video_acc'][-1]:.4f}\n"
                f"Latest Train Loss: {self.metrics_history['train_loss'][-1]:.4f}"
            )
            ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plot_dir / f'epoch_{epoch:03d}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {plot_path}")
    
    def get_best_epoch(self) -> tuple:
        """Get best epoch based on video AUC"""
        if not self.metrics_history['video_auc']:
            return None, None
        
        best_idx = np.argmax(self.metrics_history['video_auc'])
        best_epoch = self.metrics_history['epoch'][best_idx]
        best_auc = self.metrics_history['video_auc'][best_idx]
        
        return best_epoch, best_auc


# Sửa hàm train() để sử dụng logger
def train(args, model, optimizer, train_loader, valid_loader, scheduler, save_dir):
    max_accuracy = 0
    global_step = 0
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Khởi tạo logger
    logger = TrainingLogger(save_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # checkpoint
    if args.resume > -1:
        checkpoint = torch.load(os.path.join(save_dir, 'models_params_{}.tar'.format(args.resume)),
                                map_location='cuda:{}'.format(torch.cuda.current_device()))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))

    for epoch in range(args.resume+1, args.epochs):
        # train part
        print('start train mode...')
        epoch_loss = 0.0
        total_num = 0
        correct_num = 0
        model.train()

        with torch.enable_grad():
            st_time = time.time()
            for i, (inputs, labels) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs, moe_loss = model(inputs)
                ce_loss = criterion(outputs, labels)
                loss = ce_loss + 1*moe_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_num += inputs.size(0)
                correct_num += torch.sum(torch.argmax(outputs, 1) == labels).item()
                global_step += 1

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct_num / total_num

        # eval part
        print('start eval mode...')
        model.eval()
        video_predictions = []
        video_labels = []
        frame_predictions = []
        frame_labels = []

        with torch.no_grad():

            for inputs,labels in tqdm(valid_loader,total=len(valid_loader),ncols=70,leave=False,unit='step'):
                # print("\n--- Eval Step ---")
                # logging.info(inputs.shape)
                # print(inputs, labels)
                # print("-----")
                inputs = inputs.cuda()
                inputs = inputs.squeeze(0)
                labels = labels.cuda()

                inputs = inputs.view(-1, 3, inputs.shape[-2], inputs.shape[-1])
                # logging.info(inputs.shape)
                
                outputs,_ = model(inputs)
                # outputs = model(inputs)
                outputs = F.softmax(outputs, dim=-1)
                frame = outputs.shape[0]
                frame_predictions.extend(outputs[:,1].cpu().tolist())
                frame_labels.extend(labels.expand(frame).cpu().tolist())
                pre = torch.mean(outputs[:,1])
                video_predictions.append(pre.cpu().item())
                video_labels.append(labels.cpu().item())

        frame_results = cal_metrics(frame_labels, frame_predictions, threshold=0.5)
        video_results = cal_metrics(video_labels, video_predictions, threshold=0.5)
        
        # Log metrics to CSV and plot
        metrics_dict = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'video_acc': video_results.ACC,
            'video_auc': video_results.AUC,
            'video_eer': video_results.EER,
            'frame_acc': frame_results.ACC,
            'frame_auc': frame_results.AUC,
            'frame_eer': frame_results.EER,
        }
        logger.log_epoch(epoch, metrics_dict)
        
        print(f'Epoch [{epoch+1:0>3}/{args.epochs:0>3}], '
              f'V_Acc: {video_results.ACC:.2%}, V_AUC: {video_results.AUC:.4f}, '
              f'F_Acc: {frame_results.ACC:.2%}, F_AUC: {frame_results.AUC:.4f}')

        # save model
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, os.path.join(save_dir, 'models_params_{}.tar'.format(epoch)))

        if video_results.AUC > max_accuracy:
            for m in os.listdir(save_dir):
                if m.startswith('model_params_best'):
                    current_models = m
                    os.remove(os.path.join(save_dir, current_models))
            max_accuracy = video_results.AUC
            torch.save(model.state_dict(), 
                      '{}model_params_best_{:.4f}auc{:.4f}epoch{:0>3}.pkl'.format(
                          args.model_dir, video_results.ACC, video_results.AUC, epoch+1))

        scheduler.step()
    
    # Print summary
    best_epoch, best_auc = logger.get_best_epoch()
    print(f'\nTraining completed!')
    print(f'Best epoch: {best_epoch} with AUC: {best_auc:.4f}')
    print(f'Metrics saved to: {logger.csv_file}')
    print(f'Plots saved to: {logger.plot_dir}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device','-dv', type=int, default=0, help="specify which GPU to use")
    parser.add_argument('--model_dir', '-md', type=str, default='models/train')
    parser.add_argument('--resume','-rs', type=int, default=-1, help="which epoch continue to train")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--record_step', type=int, default=100, help="the iteration number to record train state")

    parser.add_argument('--batch_size','-bs', type=int, default=32)
    parser.add_argument('--learning_rate','-lr', type=float, default=3e-5)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()


    start_time = time.time()
    setup_seed(2024)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    save_dir = args.model_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)

    # logging
    if args.resume == -1:
        mode = 'w'
    else:
        mode = 'a'
    logging.basicConfig(
        filename=os.path.join(save_dir, 'train.log'),
        filemode=mode,
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    log.addHandler(handler)
    logging.info(args.model_dir)
    log.info('model dir:' + args.model_dir)

    # train_path = '/data3/law/data/FF++/c23/train'
    # valid_path = '/data3/law/data/FF++/c23/valid'


    # train_dataset = FFPP_Dataset(train_path,frame=20,phase='train')
    # valid_dataset =TestDataset(valid_path,dataset='FFPP',frame=20)

    # from dataloader import train_set, test_set
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # valid_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=True)
    from dataloaderv2 import get_dataloader
    root = "/home/toi/research/face_forgery_detection/datasets/ffpp"

    train_loader = get_dataloader(
        root,
        dataset_name="FFPP",
        split_json=os.path.join(root, "train.json"),
        batch_size=32,
        phase="train"
    )

    test_loader = get_dataloader(
        root,
        dataset_name="FFPP",
        split_json=os.path.join(root, "test.json"),
        batch_size=1,
        phase="test"
    )

    model = vit_base_patch16_224_in21k(pretrained=True,num_classes=2)
    model = model.cuda()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # special defined optim
    special_param = []
    other_param = []
    for name, param in model.named_parameters():
        if 'w_gate' in name or 'w_noise' in name:
            special_param.append(param)
        else:
            other_param.append(param)

    optimizer = optim.Adam([{'params': special_param, 'initial_lr':1e-4}, {'params': other_param, 'initial_lr': args.learning_rate}],
                           lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=args.resume)


    print('Start train process...')
    train(args, model,optimizer,train_loader,test_loader,scheduler,save_dir)
    duration = time.time()-start_time
    # print('The task of {} is completed'.format(args.description))
    # print('The best AUC is {:.2%}'.format(auc))
    print('The run time is {}h {}m'.format(int(duration//3600),int(duration%3600//60)))
