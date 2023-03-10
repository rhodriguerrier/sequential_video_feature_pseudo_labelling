import torch
import torch.nn as nn
from transformer import ViT
from mlp import FlatMLP
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pre_extract_kitchens_dataset import  EpicVariableAugFt, EpicConcatAugFt, EpicExtractedAugFt
import numpy as np
import pandas as pd
import sys
import random
import argparse
import logging


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

dataset_aug_map = {
    "transformer": EpicExtractedAugFt,
    "var_transformer": EpicVariableAugFt,
    "mlp": EpicConcatAugFt
}


class ViTModel:
    def __init__(self, lr, batch_size, num_epochs, architecture="transformer"):
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.architecture = architecture
        self.train_dataset = dataset_aug_map[architecture]("src", "train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.src_test_dataset =  dataset_aug_map[architecture]("src", "test")
        self.src_test_dataloader = DataLoader(self.src_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.trg_test_dataset = dataset_aug_map[architecture]("trg", "test")
        self.trg_test_dataloader = DataLoader(self.trg_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        if architecture == "mlp":
            self.model = FlatMLP()
        else:
            self.model = ViT()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.000001)
        self.lr_sched = optim.lr_scheduler.MultiStepLR(self.optim, [10])
        self.ce_loss = nn.CrossEntropyLoss()


    def train_mlp(self):
        self.test_mlp()
        self.optim.zero_grad()
        for epoch in range(self.num_epochs):
            logging.info(f"Epoch: {epoch}")
            self.model.train()
            for idx, (src_labels, rgb, flow, rgb_idx, flow_idx, indices) in enumerate(self.train_dataloader):
                rgb_output, flow_output = self.model(rgb, flow)
                train_loss = self.ce_loss(rgb_output+flow_output, src_labels.long())
                logging.info(f"Loss = {train_loss}")
                train_loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            self.test_mlp()
            if epoch == self.num_epochs-1:
                torch.save(self.model.state_dict(), f"./trained_models/self_attention_models/mlp_src_train_lr_{'_'.join(str(self.lr).split('.'))}_bs_48_with_aug.pt")


    def test_mlp(self):
        self.model.eval()
        total, total_correct = 0, 0
        for (src_labels, rgb_ft, flow_ft, indices) in self.src_test_dataloader:
            total += src_labels.size(0)
            rgb_output, flow_output = self.model(rgb_ft, flow_ft)
            predictions = torch.argmax(F.softmax(rgb_output+flow_output, dim=1), dim=1)
            mask = (src_labels.long() == predictions)
            total_correct += predictions[mask].size(0)
        logging.info(f"Source Test Label Accuracy = {(total_correct/total)*100.0}%")

        total, total_correct = 0, 0
        for (src_labels, rgb_ft, flow_ft, indices) in self.trg_test_dataloader:
            total += src_labels.size(0)
            rgb_output, flow_output = self.model(rgb_ft, flow_ft)
            predictions = torch.argmax(F.softmax(rgb_output+flow_output, dim=1), dim=1)
            mask = (src_labels.long() == predictions)
            total_correct += predictions[mask].size(0)
        logging.info(f"Target Test Label Accuracy = {(total_correct/total)*100.0}%")
        

    def train_transformer(self):
        self.test_transformer()
        self.optim.zero_grad()
        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch}")
            self.model.train()
            for idx, (src_labels, rgb_ft, flow_ft, rgb_idx, flow_idx, indices) in enumerate(self.train_dataloader):
                reshaped_rgb_ft = reshape_feature_tensors(rgb_ft)
                reshaped_flow_ft = reshape_feature_tensors(flow_ft)
                output = self.model(reshaped_rgb_ft, reshaped_flow_ft)
                train_loss = self.ce_loss(output, src_labels.long())
                logging.info(f"Loss = {train_loss}")
                train_loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            self.lr_sched.step()
            self.test_transformer()
            if epoch == self.num_epochs-1:
                torch.save(self.model.state_dict(), f"./trained_models/self_attention_models/transformer_src_train_lr_{'_'.join(str(self.lr).split('.'))}_bs_48_with_aug.pt")


    def test_transformer(self):
        self.model.eval()
        total, total_correct = 0, 0
        for (src_labels, rgb_ft, flow_ft, indices) in self.src_test_dataloader:
            total += src_labels.size(0)
            reshaped_rgb_ft = reshape_feature_tensors(rgb_ft)
            reshaped_flow_ft = reshape_feature_tensors(flow_ft)
            output = self.model(reshaped_rgb_ft, reshaped_flow_ft)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            mask = (src_labels.long() == predictions)
            total_correct += predictions[mask].size(0)
        logging.info(f"Source Test Label Accuracy = {(total_correct/total)*100.0}%")

        total, total_correct = 0, 0
        for (src_labels, rgb_ft, flow_ft, indices) in self.trg_test_dataloader:
            total += src_labels.size(0)
            reshaped_rgb_ft = reshape_feature_tensors(rgb_ft)
            reshaped_flow_ft = reshape_feature_tensors(flow_ft)
            output = self.model(reshaped_rgb_ft, reshaped_flow_ft)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            mask = (src_labels.long() == predictions)
            total_correct += predictions[mask].size(0)
        logging.info(f"Target Test Label Accuracy = {(total_correct/total)*100.0}%")


    def train_variable_transformer(self):
        self.test_variable_transformer()
        self.optim.zero_grad()
        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch}")
            self.model.train()
            for idx, (src_labels, indices) in enumerate(self.train_dataloader):
                rgb_ft, flow_ft, attn_mask = self.train_dataset.get_padded_batch(indices["index"], torch.max(indices['len'], dim=0)[0].item())
                output = self.model(rgb_ft, flow_ft, attn_mask)
                formatted_labels = torch.tensor([int(label_str.split("_")[0]) for label_str in src_labels])
                train_loss = self.ce_loss(output, formatted_labels.long())
                logging.info(f"Loss = {train_loss}")
                train_loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            self.lr_sched.step()
            self.test_variable_transformer()
            if epoch == self.num_epochs-1:
                torch.save(self.model.state_dict(), f"./trained_models/self_attention_models/var_transformer_src_train_lr_{'_'.join(str(self.lr).split('.'))}_bs_48_with_aug.pt")


    def test_variable_transformer(self):
        self.model.eval()
        total, total_correct = 0, 0
        for (src_labels, indices) in self.src_test_dataloader:
            total += len(src_labels)
            rgb_ft, flow_ft, attn_mask = self.src_test_dataset.get_padded_batch(indices["index"], torch.max(indices['len'], dim=0)[0].item())
            output = self.model(rgb_ft, flow_ft, attn_mask)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            mask = (torch.tensor([int(label_str.split("_")[0]) for label_str in src_labels]).long() == predictions)
            total_correct += predictions[mask].size(0)
        logging.info(f"Source Test Label Accuracy = {(total_correct/total)*100.0}%")

        total, total_correct = 0, 0
        for (src_labels, indices) in self.trg_test_dataloader:
            total += len(src_labels)
            rgb_ft, flow_ft, attn_mask = self.trg_test_dataset.get_padded_batch(indices["index"], torch.max(indices['len'], dim=0)[0].item())
            output = self.model(rgb_ft, flow_ft, attn_mask)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            formatted_labels = torch.tensor([int(label_str.split("_")[0]) for label_str in src_labels])
            mask = (formatted_labels.long() == predictions)
            total_correct += predictions[mask].size(0)
        logging.info(f"Target Test Label Accuracy = {(total_correct/total)*100.0}%")


def reshape_feature_tensors(original):
    cat_ls = []
    for sample in original:
        cat_ls.append(torch.unsqueeze(sample, dim=1))
    return torch.cat(cat_ls, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epic Kitchens Feature Extraction")
    parser.add_argument("--src_domain_id", action="store", dest="src_domain_id", default="D1")
    parser.add_argument("--trg_domain_id", action="store", dest="trg_domain_id", default="D2")
    parser.add_argument("--epochs", action="store", dest="epochs", default="20")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="48")
    parser.add_argument("--lr", action="store", dest="lr", default="0.0005")
    parser.add_argument("--architecture", action="store", dest="architecture", default="transformer")
    args = parser.parse_args()
    logging.basicConfig(
        filename=f"{args.architecture}_training_lr_0_0005_with_adam_aug.log",
        filemode="a",
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )

    model_runner = ViTModel(
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        num_epochs=int(args.epochs),
        architecture=args.architecture
    )
    if args.architecture == "mlp":
        model_runner.train_mlp()
    elif args.architecture == "transformer":
        model_runner.train_transformer()
    elif args.architecture == "var_transformer":
        model_runner.train_variable_transformer()
