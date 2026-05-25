import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
torch.manual_seed(42)  # Set seed
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from utils import bootstrap, get_eval_metrics, seed_torch
from datasets import Sequential_Generic_MIL_Dataset

import sys
sys.path.append("/home/bui/continual_learning/CATE_ACCESS/src/")
from prompts_zeroshot import brca_prompts, rcc_prompts, nsclc_prompts, esca_prompts, tgct_prompts, cesc_prompts
device = 'cuda:0'
titan_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
titan_model = titan_model.to(device)

_, TEMPLATES = brca_prompts()
CLASS_PROMPTS = []

print("Getting Prompt Prototypes ...")
for prompts in [brca_prompts, rcc_prompts, nsclc_prompts, esca_prompts, tgct_prompts, cesc_prompts]:
    class_prompts, _ = prompts()
    CLASS_PROMPTS.extend(class_prompts)

with torch.autocast('cuda', torch.float16), torch.inference_mode():
    classifier = titan_model.zero_shot_classifier(CLASS_PROMPTS, TEMPLATES, device=device)

dict_classes = {
    0: [0, 1],
    1: [2, 4],
    2: [5, 6],
    3: [7, 8],
    4: [9, 10],
    5: [11, 12]
}

"""
Script to finetune TITAN on a dummy dataset. Dataset class needs to be adapted to a custom dataset and task.
"""

MAX_NUM_PATCHES = 10000

class CustomSequential(nn.Module):
    def __init__(self, model, mlp):
        super(CustomSequential, self).__init__()
        self.backbone = model.vision_encoder
        self.mlp = mlp

    def forward(self, features, coords, ps):
        x = self.backbone(features, coords, ps)
        x = self.mlp(x)
        return x

def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.0, out_dim=None, end_with_fc=True):
    layers = []
    if len(hid_dims) > 0:
        for hid_dim in hid_dims:
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            in_dim = hid_dim
    layers.append(nn.Linear(in_dim, out_dim))
    if not end_with_fc:
        layers.append(act)
        layers.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*layers)
    return mlp


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    """Copied from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/scheduler.py
    """
    def _warmup_lr(base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length
    
    def _assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = new_lr * param_group["lr_scale"]
            else:
                param_group["lr"] = new_lr
    
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        _assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        """
        Args:
            patience (int): How long to wait after the last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        # Check if the new loss is an improvement
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_weights = model.state_dict()
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.best_model_weights = model.state_dict()

def eval(test_loader, model, dict_class, device, prefix, save_location, **kwargs):
    preds_all = []
    probs_all = []
    targets_all = []
    K = 400
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader):
            label = torch.Tensor([dict_class[int(label[0])]])
            features = features.to(device)
            coords = coords.long().to(device)

            indices = torch.randperm(features.shape[0])[:K]

            features = features[indices, :]
            coords = coords[indices, :]

            try:
                logits = model(features, coords, torch.tensor(1024).int().to(device), **kwargs)
            except:
                model.cpu()
                logits = model(features, coords, torch.tensor(1024).int().cpu(), **kwargs)
                model.to(device)
            
            logits = logits.float()
            preds = logits.argmax(1)

            probs = nn.functional.softmax(logits, dim=1)
            roc_kwargs = {"multi_class": "ovo", "average": "macro"}

            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

        preds_all = np.concatenate(preds_all)
        probs_all = np.concatenate(probs_all)
        targets_all = np.concatenate(targets_all)

    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs=roc_kwargs, prefix=prefix)

    return eval_metrics, preds_all, targets_all

def forgetting(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)

def backward_transfer(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, 0)

    parser = argparse.ArgumentParser(description="Finetune TITAN")
    
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./logs")

    args = parser.parse_args()
    
    num_tasks = 6
    dict_convert_class = {
        0: {0:0, 1:1},
        1: {0:2, 1:3, 2:4},
        2: {0:5, 1:6},
        3: {0:7, 1:8},
        4: {0:9, 1:10},
        5: {0:11, 1:12}
    }
    seq_dataset = Sequential_Generic_MIL_Dataset()

    overall_accs = []
    list_num_tasks = [1, 2, 3, 4, 5, 6]
    list_num_classes = [2, 5, 7, 9, 11, 13]
    mACCs_all_folds = []
    fgt_all_folds = []
    bwt_all_folds = []

    f = open("acc_per_task_MergeSlide.txt", "w")
    for fold_id in range(0, 10):
        fold = "fold_" + str(fold_id)

        task_models = [
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_0.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_1.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_2.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_3.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_4.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_5.pt"
        ]

        print("Testing", fold)
        mean_ACCs = []
        acc_per_task_all_tasks = []
        for seq_task in tqdm(list_num_tasks):
            seed_torch(device, 0)
            acc_per_task = [0 for t in range(0, seq_task)]
            num_tasks = seq_task
            num_class = list_num_classes[seq_task-1]
            merge_model_path = "/home/bui/continual_learning/WSIModelMerging/fusion_bench/method/opcm_no_rank_new/merged_weight_opcm_random_sampling_{}".format(fold) + \
                                "/" + "merged_weight_opcm_random_sampling_{}_task_{}".format(fold, seq_task-1) + ".pth"

            # load model from huggingface
            base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)    
            base_model = base_model.to(device)
            # load merged weight
            base_model.vision_encoder.load_state_dict(torch.load(merge_model_path))

            # add mlp head for finetuning
            mlp = nn.Linear(768, num_class).to(device)
            mlp.weight.data.normal_(mean=0.0, std=0.01)
            mlp.bias.data.zero_()
            model = CustomSequential(base_model, mlp)
            # load MLP
            mlp_task_weights = [torch.load(task_models[task_id]) for task_id in range(num_tasks)]
            
            for i in range(len(mlp_task_weights)):
                mlp_task_weights[i] = {k.split('mlp.')[-1]:mlp_task_weights[i][k] for k in list(mlp_task_weights[i].keys())[-2:]}
            
            merge_mlp_data = dict()
            merge_mlp_data['weight'] = torch.cat([data['weight'] for data in mlp_task_weights])
            merge_mlp_data['bias'] = torch.cat([data['bias'] for data in mlp_task_weights])

            model.mlp.load_state_dict(merge_mlp_data)

            num_total = 0.
            num_correct = 0.
            for task_id in range(num_tasks):
                # print("TASK", task_id)
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
                model.eval()
                results, preds_all, targets_all = eval(test_loader, model, dict_convert_class[task_id], device, prefix="", save_location=None)
                
                num_correct += sum(preds_all == targets_all)
                num_total += len(test_loader)
                acc_per_task[task_id] = sum(preds_all == targets_all) / len(targets_all)
            
            f.write("acc_fold_" + str(fold_id) + str(acc_per_task) + "\n")
            acc_per_task_all_tasks.append(acc_per_task)
        
        fgt = forgetting(acc_per_task_all_tasks)
        bwt = backward_transfer(acc_per_task_all_tasks)
        fgt_all_folds.append(fgt)
        bwt_all_folds.append(bwt)
        print(acc_per_task_all_tasks[-1])
        
        print(acc_per_task_all_tasks)
        mACC = 0.
        for i, seq in enumerate(acc_per_task_all_tasks):
            mACC += (np.sum(seq) / list_num_tasks[i])
        mACC /= len(acc_per_task_all_tasks)
        mACCs_all_folds.append(mACC)

    print("mACC", np.mean(mACCs_all_folds), "std", np.std(mACCs_all_folds))
    print("BWT", np.mean(bwt_all_folds), "std", np.std(bwt_all_folds))
    print("FGT", np.mean(fgt_all_folds), "std", np.std(fgt_all_folds))
    f.close()