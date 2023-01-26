import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from kitchens_dataset import EpicMultiModalTestDataset, EpicMultiModalSrcFreeWithPseudoLabels, SequentialMultiModalKitchens
from pytorch_i3d import InceptionI3d
import videotransforms
import numpy as np
import pickle
import argparse
import logging
import math


class EpicKitchensI3D:
    def __init__(
            self,
            num_epochs,
            init_lr,
            batch_size,
            src_labels_path,
            src_domain_id,
            pseudo_type,
    ):
        self.num_epochs = num_epochs
        self.src_domain_id = src_domain_id
        self.lr = init_lr
        self.batch_size = batch_size
        self.src_labels_path = src_labels_path
        self.pseudo_type = pseudo_type
        
        # Flow model
        self.flow_model = InceptionI3d(8, in_channels=2)
        self.flow_model.cuda()
        self.flow_model = nn.DataParallel(self.flow_model)
        self.flow_model.load_state_dict(torch.load(f'./trained_models/flow_mm_{src_domain_id}_train_100_epochs.pt'))
        
        # RGB model
        self.rgb_model = InceptionI3d(8, in_channels=3)
        self.rgb_model.cuda()
        self.rgb_model = nn.DataParallel(self.rgb_model)
        self.rgb_model.load_state_dict(torch.load(f'./trained_models/rgb_mm_{src_domain_id}_train_100_epochs.pt'))
        
        self.flow_optim = optim.SGD(self.flow_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0000001)
        self.rgb_optim = optim.SGD(self.rgb_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0000001)
        self.flow_lr_sched = optim.lr_scheduler.MultiStepLR(self.flow_optim, [10, 20])
        self.rgb_lr_sched = optim.lr_scheduler.MultiStepLR(self.rgb_optim, [10, 20])

        self.num_steps_per_update = 6
        self.train_transforms = transforms.Compose([
            videotransforms.RandomCrop(224),
            videotransforms.RandomHorizontalFlip()
        ])
        self.test_transforms = transforms.Compose([
            videotransforms.CenterCrop(224)
        ])
        self.ce_loss = nn.CrossEntropyLoss()


    def test_accuracy(
            self,
            trg_label_path
    ):
        self.flow_model.eval()
        self.rgb_model.eval()
        total, total_correct = 0, 0
        logging.info(f"Starting Extraction of Test Pseudo Labels")
        class_dataset = EpicMultiModalTestDataset(
            labels_path=trg_label_path,
            transforms=self.test_transforms
        )
        dataloader = DataLoader(class_dataset, batch_size=1, shuffle=False, num_workers=0)
        for (labels, rgb_seg_img_inputs, flow_seg_img_inputs, seq_window_start_ratios) in dataloader:
            total += 1
            # RGB
            rgb_seq_action_concat = torch.tensor([])
            for rgb_img_input in rgb_seg_img_inputs:
                rgb_seq_action_concat = torch.cat((rgb_seq_action_concat, rgb_img_input))
            rgb_inputs = torch.tensor(rgb_seq_action_concat).float()
            rgb_logits = self.rgb_model(rgb_inputs)
            rgb_logits_reshaped = torch.reshape(rgb_logits, (rgb_logits.size(0), rgb_logits.size(1))).detach().cpu()
            rgb_logits_avg = torch.mean(rgb_logits, dim=0)

            # Flow
            flow_seq_action_concat = torch.tensor([])
            for flow_img_input in flow_seg_img_inputs:
                flow_seq_action_concat = torch.cat((flow_seq_action_concat, flow_img_input))
            flow_inputs = torch.tensor(flow_seq_action_concat).float()
            flow_logits = self.flow_model(flow_inputs)
            flow_logits_reshaped = torch.reshape(flow_logits, (flow_logits.size(0), flow_logits.size(1))).detach().cpu()
            flow_logits_avg = torch.mean(flow_logits, dim=0)

            # Fused Averaged Predictions & Classify
            fused_confidence = F.softmax(rgb_logits_avg + flow_logits_avg, dim=0)
            pseudo_label = torch.argmax(fused_confidence, dim=0).item()
            if pseudo_label == labels[0].item():
                total_correct += 1
        logging.info(f"Accuracy = {(total_correct/total)*100}%")


    def extract_max_window_pseudo_labels(
            self,
            prev_trg_label_path,
            new_trg_label_path
    ):
        self.rgb_model.eval()
        self.flow_model.eval()
        trg_label_df = load_pickle_data(prev_trg_label_path)
        trg_label_df["pseudo_label"] = np.nan
        trg_label_df["confidence"] = np.nan
        for i in [6, 0, 1, 2, 3, 4, 5, 7]:
            logging.info(f"Starting Extraction of Pseudo Labels for Class {i}...")
            class_dataset = SequentialMultiModalKitchens(
                labels_path=prev_trg_label_path,
                class_num=i,
                temporal_window=16,
                step=2,
                sequential_overlap=16,
                transforms=self.train_transforms
            )
            dataloader = DataLoader(class_dataset, batch_size=1, shuffle=False, num_workers=0)
            for (labels, rgb_seq_img_inputs, flow_seq_img_inputs) in dataloader:
                logging.info(f"RGB: {len(rgb_seq_img_inputs)}, Flow: {len(flow_seq_img_inputs)}")
                # Flow
                step = 6
                flow_instance_seq_logits = torch.tensor([])
                counter = 0
                for j in range(0, len(flow_seq_img_inputs), step):
                    if j + step > len(flow_seq_img_inputs):
                        temp_inputs = torch.tensor(torch.cat(flow_seq_img_inputs[j:])).float().cuda()
                    else:
                        temp_inputs = torch.tensor(torch.cat(flow_seq_img_inputs[j:j+step])).float().cuda()
                    flow_logits = self.flow_model(temp_inputs)
                    flow_logits_reshaped = torch.reshape(flow_logits, (flow_logits.size(0), flow_logits.size(1))).detach().cpu()
                    flow_instance_seq_logits = torch.cat((flow_instance_seq_logits, flow_logits_reshaped))
                # RGB
                step = 6
                rgb_instance_seq_logits = torch.tensor([])
                counter = 0
                for j in range(0, len(rgb_seq_img_inputs), step):
                    if j + step > len(rgb_seq_img_inputs):
                        temp_inputs = torch.tensor(torch.cat(rgb_seq_img_inputs[j:])).float().cuda()
                    else:
                        temp_inputs = torch.tensor(torch.cat(rgb_seq_img_inputs[j:j+step])).float().cuda()
                    rgb_logits = self.rgb_model(temp_inputs)
                    rgb_logits_reshaped = torch.reshape(rgb_logits, (rgb_logits.size(0), rgb_logits.size(1))).detach().cpu()
                    rgb_instance_seq_logits = torch.cat((rgb_instance_seq_logits, rgb_logits_reshaped))

                # Fuse logits, softmax and find max window
                fused_logits = flow_instance_seq_logits + rgb_instance_seq_logits
                if self.pseudo_type == "max_seg":
                    pseudo_label, confidence = self.extract_max_window_pseudo_label(fused_logits)
                elif self.pseudo_type == "max_agree":
                    pseudo_label, confidence = self.extract_max_agree_pseudo_label(fused_logits)
                trg_label_df.loc[trg_label_df["uid"] == int(labels[0].split("_")[1]), "pseudo_label"] = pseudo_label
                trg_label_df.loc[trg_label_df["uid"] == int(labels[0].split("_")[1]), "confidence"] = confidence
        trg_label_df.to_pickle(new_trg_label_path)
        logging.info(f"Extracted Pseudo Labels saved at {new_trg_label_path}")


    def extract_max_window_pseudo_label(self, fused_logits):
        instance_seq_pred = F.softmax(fused_logits, dim=1)
        max_val_at = torch.argmax(instance_seq_pred)
        row = math.floor(max_val_at / instance_seq_pred.size(1))
        pseudo_label = (max_val_at - (instance_seq_pred.size(1)*row))
        confidence = self.ce_loss(fused_logits[row], pseudo_label)
        return pseudo_label.item(), confidence.item()


    def extract_max_agree_pseudo_label(self, fused_logits):
        instance_seq_pred = F.softmax(fused_logits, dim=1)
        segment_pseudos = torch.argmax(instance_seq_pred, dim=1)
        unique_labels, counts = segment_pseudos.unique(return_counts=True)
        max_agree_pseudo = unique_labels[torch.argmax(counts, dim=0).item()].item()
        max_agree_logits = fused_logits[torch.squeeze(torch.nonzero(segment_pseudos == max_agree_pseudo), dim=1)]
        confidence = self.ce_loss(torch.mean(max_agree_logits, dim=0), torch.tensor(max_agree_pseudo)).item()
        return max_agree_pseudo, confidence

 
    def train_with_pseudos(self, prev_trg_pseudo_labels_path, trg_pseudo_sample_rate):
        for epoch in range(0, self.num_epochs):
            counter = 0
            # Extract pseudo labels for epoch training
            logging.info(f"Epoch {epoch}")
            trg_pseudo_labels_path = f"./label_lookup/{self.pseudo_type}_train_pseudo_labels_{epoch}.pkl"
            logging.info(f"Extracting {self.pseudo_type} pseudo labels...")
            self.extract_max_window_pseudo_labels(prev_trg_pseudo_labels_path, trg_pseudo_labels_path)

            # Set networks to train and create dataset with new pseudo labels
            self.rgb_model.train()
            self.flow_model.train()
            prev_trg_pseudo_labels_path = trg_pseudo_labels_path
            logging.info(f"Using labels from {trg_pseudo_labels_path}...")
            train_dataset = EpicMultiModalSrcFreeWithPseudoLabels(
                trg_pseudo_labels_path,
                trg_pseudo_sample_rate,
                transforms=self.train_transforms
            )
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            self.rgb_optim.zero_grad()
            self.flow_optim.zero_grad()
            logging.info("Beginning training...")
            sum_loss = 0.0
            num_batches = len(train_dataloader)
            on_final_batch = False
            for batch_idx, (train_labels, rgb_inputs, flow_inputs, narration_ids) in enumerate(train_dataloader):
                counter += 1
                if not on_final_batch:
                    if (counter == 1 and (num_batches - batch_idx) < self.num_steps_per_update):
                        on_final_batch = True
                        update_div = num_batches - batch_idx
                    else:
                        update_div = self.num_steps_per_update

                #Flow
                flow_inputs = torch.tensor(flow_inputs).float()
                flow_inputs = Variable(flow_inputs.cuda())
                flow_output = self.flow_model(flow_inputs)

                #RGB
                rgb_inputs = torch.tensor(rgb_inputs).float()
                rgb_inputs = Variable(rgb_inputs.cuda())
                rgb_output = self.rgb_model(rgb_inputs)

                # Calculate joint loss
                train_labels = Variable(train_labels.cuda())
                fused_output = torch.reshape(rgb_output, (rgb_output.size()[0], rgb_output.size()[1])) + torch.reshape(flow_output, (flow_output.size()[0], flow_output.size()[1]))
                train_ce_loss = self.ce_loss(fused_output, train_labels.long())
                loss = train_ce_loss / update_div
                sum_loss += loss
                loss.backward()
                logging.info(f"Batch Number = {batch_idx}")

                # Update network with accumulated batch of 48
                if counter == update_div:
                    counter = 0
                    logging.info(f"Completing Step, Loss: {sum_loss}")
                    sum_loss = 0.0
                    self.rgb_optim.step()
                    self.flow_optim.step()
                    self.rgb_optim.zero_grad()
                    self.flow_optim.zero_grad()
            self.rgb_lr_sched.step()
            self.flow_lr_sched.step()

            # Test updated network on test data for accuracy
            logging.info(f"Epoch {epoch} same single batch training finished")
            logging.info("Evaluating test data")
            self.test_accuracy("./label_lookup/D2_test.pkl")
            logging.info("---------------------------------")
            if epoch  == self.num_epochs-1:
                self.extract_max_window_pseudo_labels(prev_trg_pseudo_labels_path, f"./label_lookup/{self.pseudo_type}_train_pseudo_labels_final.pkl")
            torch.save(self.flow_model.state_dict(), f"./trained_models/flow_{self.pseudo_type}_src_free_train_epoch_{epoch}.pt")
            torch.save(self.rgb_model.state_dict(), f"./trained_models/rgb_{self.pseudo_type}_src_free_train_epoch_{epoch}.pt")


def load_pickle_data(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epic Kitchens Feature Extraction")
    parser.add_argument("--src_domain_id", action="store", dest="src_domain_id", default="D1")
    parser.add_argument("--trg_domain_id", action="store", dest="trg_domain_id", default="D2")
    parser.add_argument("--epochs", action="store", dest="epochs", default="60")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="8")
    parser.add_argument("--pseudo_sample_rate", action="store", dest="pseudo_sample_rate", default="0.2")
    parser.add_argument("--pseudo_type", action="store", dest="pseudo_type", default="max_seg")
    args = parser.parse_args()
    logging_filename = f"train_{args.trg_domain_id}_{args.pseudo_type}.log"
    logging.basicConfig(
        filename=logging_filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    model = EpicKitchensI3D(
        num_epochs=int(args.epochs),
        init_lr=0.002,
        batch_size=int(args.batch_size),
        src_labels_path=f"./label_lookup/{args.src_domain_id}_train.pkl",
        src_domain_id=args.src_domain_id,
        pseudo_type=args.pseudo_type
    )
    logging.info("Evaluating test data on Source Only Model")
    model.test_accuracy("./label_lookup/D2_test.pkl")
    model.train_with_pseudos(
        f"./label_lookup/{args.trg_domain_id}_train.pkl",
        float(args.pseudo_sample_rate)
    )
