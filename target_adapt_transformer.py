import torch
import torch.nn as nn
from transformer import ViT
from mlp import FlatMLP
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pre_extract_kitchens_dataset import EpicExtractedAugFtWithPseudos, EpicExtractedAugFt, EpicVariableAugFt, EpicVariableAugFtWithPseudos, EpicConcatAugFt, EpicConcatAugFtWithPseudos
import numpy as np
import pandas as pd
import sys
import random
import argparse
import logging


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

dataset_map = {
    "transformer": EpicExtractedAugFt,
    "var_transformer": EpicVariableAugFt,
    "mlp": EpicConcatAugFt
}


class ModelAdapter:
    def __init__(self, lr, batch_size, num_epochs, architecture="transformer", add_one_noisy=False):
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.architecture = architecture
        self.add_one_noisy = add_one_noisy
        if self.architecture == "mlp":
            self.model = FlatMLP()
        else:
            self.model = ViT()
        self.model.load_state_dict(torch.load(f"./trained_models/self_attention_models/{self.architecture}_src_train_lr_0_0005_bs_48_with_aug.pt"))
        self.train_dataset = dataset_map[architecture]("trg", "train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.000001)
        self.lr_sched = optim.lr_scheduler.MultiStepLR(self.optim, [10, 20])
        self.ce_loss = nn.CrossEntropyLoss()


    def extract_pseudo_labels(self):
        trg_pseudo_labels = pd.DataFrame(columns=["label_idx", "actual_label", "pseudo_label", "confidence", "rgb_segs", "flow_segs"])
        for (labels, rgb_ft, flow_ft, rgb_seg_idx, flow_seg_idx, indices) in self.train_dataloader:
            if self.architecture == "transformer":
                output = self.model(reshape_feature_tensors(rgb_ft), reshape_feature_tensors(flow_ft))
            else:
                rgb_output, flow_output = self.model(rgb_ft, flow_ft)
                output = rgb_output + flow_output
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            for i in range(labels.size(0)):
                rgb_segs = ",".join([str(val.item()) for val in rgb_seg_idx[i]])
                flow_segs = ",".join([str(val.item()) for val in flow_seg_idx[i]])
                trg_pseudo_labels = pd.concat((
                    trg_pseudo_labels,
                    pd.DataFrame(
                        [[indices[i].item(), labels[i].item(), predictions[i].item(), self.ce_loss(output[i], labels[i].long()).item(), rgb_segs, flow_segs]],
                        columns=["label_idx", "actual_label", "pseudo_label", "confidence", "rgb_segs", "flow_segs"]
                    )
                ))
        return trg_pseudo_labels

    
    def extract_variable_pseudo_labels(self):
        self.model.eval()
        trg_pseudo_labels = pd.DataFrame(columns=["label_id", "actual_label", "pseudo_label", "confidence"])
        total, total_correct = 0, 0
        trg_train_dataset = EpicVariableAugFt("trg", "train")
        trg_train_dataloader = DataLoader(trg_train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        for (src_labels, indices) in trg_train_dataloader:
            total += len(src_labels)
            rgb_ft, flow_ft, attn_mask = trg_train_dataset.get_padded_batch(indices["index"], torch.max(indices['len'], dim=0)[0].item())
            output = self.model(rgb_ft, flow_ft, attn_mask)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            formatted_labels = torch.tensor([int(label_str.split("_")[0]) for label_str in src_labels])
            mask = (formatted_labels.long() == predictions)
            total_correct += predictions[mask].size(0)
            for i in range(formatted_labels.size(0)):
                trg_pseudo_labels = pd.concat((
                    trg_pseudo_labels,
                    pd.DataFrame(
                        [[src_labels[i], formatted_labels[i].item(), predictions[i].item(), self.ce_loss(output[i], formatted_labels[i]).item()]],
                        columns=["label_id", "actual_label", "pseudo_label", "confidence"]
                    )
                ))
        return trg_pseudo_labels


    def adapt_var_transformer(self):
        self.test_target_var_transformer()
        self.optim.zero_grad()
        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch}")
            pseudo_label_df = self.extract_variable_pseudo_labels()
            pseudo_label_df.to_pickle(f"./var_transformer_train_aug_pseudo_labels_lr_{'_'.join(str(self.lr).split('.'))}_sample_20_epoch_{epoch}.pkl")
            self.model.train()
            adapt_dataset = EpicVariableAugFtWithPseudos("trg", "train", pseudo_label_df, 0.2)
            adapt_dataloader = DataLoader(adapt_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            for idx, (labels, info) in enumerate(adapt_dataloader):
                rgb_ft, flow_ft, attn_mask = adapt_dataset.get_padded_batch(info["index"], torch.max(info["len"]).item())
                output = self.model(rgb_ft, flow_ft, attn_mask)
                train_loss = self.ce_loss(output, labels.long())
                logging.info(f"Loss = {train_loss}")
                train_loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            self.lr_sched.step()
            self.test_target_var_transformer()


    def test_target_var_transformer(self):
        self.model.eval()
        total, total_correct = 0, 0
        trg_test_dataset = EpicVariableAugFt("trg", "test")
        trg_test_dataloader = DataLoader(trg_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        for idx, (labels, info) in enumerate(trg_test_dataloader):
            total += len(labels)
            rgb_ft, flow_ft, attn_mask = trg_test_dataset.get_padded_batch(info["index"], torch.max(info["len"]).item())
            output = self.model(rgb_ft, flow_ft, attn_mask)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            formatted_labels = torch.tensor([int(label_str.split("_")[0]) for label_str in labels])
            mask = (formatted_labels.long() == predictions)
            total_correct += predictions[mask].size(0)
        logging.info(f"Target Test Label Accuracy = {(total_correct/total)*100}")


    def adapt_mlp(self):
        self.test_target_mlp()
        self.optim.zero_grad()
        for epoch in range(self.num_epochs):
            pseudo_label_df = self.extract_pseudo_labels()
            pseudo_label_df.to_pickle(f"./mlp_train_aug_pseudo_labels_lr_{'_'.join(str(self.lr).split('.'))}_sample_20_epoch_{epoch}.pkl")
            self.model.train()
            adapt_dataset = EpicConcatAugFtWithPseudos("trg", "train", pseudo_label_df, 0.2, self.add_one_noisy)
            adapt_dataloader = DataLoader(adapt_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            for (labels, rgb_ft, flow_ft) in adapt_dataloader:
                rgb_output, flow_output = self.model(rgb_ft, flow_ft)
                train_loss = self.ce_loss(rgb_output+flow_output, labels.long())
                logging.info(f"Loss = {train_loss}")
                train_loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            self.lr_sched.step()
            self.test_target_mlp()


    def test_target_mlp(self):
        self.model.eval()
        total, total_correct = 0, 0
        trg_test_dataset = EpicConcatAugFt("trg", "test")
        trg_test_dataloader = DataLoader(trg_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        for (labels, rgb_ft, flow_ft, indices) in trg_test_dataloader:
            total += labels.size(0)
            rgb_output, flow_output = self.model(rgb_ft, flow_ft)
            predictions = torch.argmax(F.softmax(rgb_output+flow_output, dim=1), dim=1)
            mask = (labels.long() == predictions)
            total_correct += predictions[mask].size(0)
        logging.info(f"Target Test Label Accuracy = {(total_correct/total)*100}")


    def adapt_transformer(self):
        print(f"Adding one noisy label = {self.add_one_noisy}")
        self.test_target_transformer()
        self.optim.zero_grad()
        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch}")
            pseudo_label_df = self.extract_pseudo_labels()
            pseudo_label_df.to_pickle(f"./transformer_train_aug_pseudo_labels_lr_{'_'.join(str(self.lr).split('.'))}_sample_20_epoch_{epoch}.pkl")
            self.model.train()
            adapt_dataset = EpicExtractedAugFtWithPseudos("trg", "train", pseudo_label_df, 0.2, self.add_one_noisy)
            adapt_dataloader = DataLoader(adapt_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            for (labels, rgb_ft, flow_ft) in adapt_dataloader:
                reshaped_rgb_ft = reshape_feature_tensors(rgb_ft)
                reshaped_flow_ft = reshape_feature_tensors(flow_ft)
                output = self.model(reshaped_rgb_ft, reshaped_flow_ft)
                train_loss = self.ce_loss(output, labels.long())
                logging.info(f"Loss = {train_loss}")
                train_loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            self.lr_sched.step()
            self.test_target_transformer()


    def test_target_transformer(self):
        self.model.eval()
        total, total_correct = 0, 0
        trg_test_dataset = EpicExtractedAugFt("trg", "test")
        trg_test_dataloader = DataLoader(trg_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        for (src_labels, rgb_ft, flow_ft, indices) in trg_test_dataloader:
            total += src_labels.size(0)
            reshaped_rgb_ft = reshape_feature_tensors(rgb_ft)
            reshaped_flow_ft = reshape_feature_tensors(flow_ft)
            output = self.model(reshaped_rgb_ft, reshaped_flow_ft)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            mask = (src_labels.long() == predictions)
            total_correct += predictions[mask].size(0)
        logging.info(f"Target Test Label Accuracy = {(total_correct/total)*100.0}%")


def reshape_feature_tensors(original):
    cat_ls = []
    for sample in original:
        cat_ls.append(torch.unsqueeze(sample, dim=1))
    return torch.cat(cat_ls, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epic Kitchens Feature Extraction")
    parser.add_argument("--epochs", action="store", dest="epochs", default="30")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="48")
    parser.add_argument("--lr", action="store", dest="lr", default="0.00007")
    parser.add_argument("--architecture", action="store", dest="architecture", default="transformer")
    parser.add_argument("--add_one_noisy", action="store_true", dest="add_one_noisy")
    parser.set_defaults(add_one_noisy=False)
    args = parser.parse_args()

    logging.basicConfig(
        filename=f"{args.architecture}_adapt_lr_0_00007_20_with_aug_no_noisy_label.log",
        filemode="a",
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )

    model_adapter = ModelAdapter(
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        num_epochs=int(args.epochs),
        architecture=args.architecture,
        add_one_noisy=args.add_one_noisy
    )
 
    if args.architecture == "var_transformer":
        model_adapter.adapt_var_transformer()
    elif args.architecture == "transformer":
        model_adapter.adapt_transformer()
    elif args.architecture == "mlp":
        model_adapter.adapt_mlp()
