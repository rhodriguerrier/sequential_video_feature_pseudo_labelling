import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from kitchens_dataset import EpicMultiModalTestDataset, EpicMultiModalDataset, EpicMultiModalSrcFreeWithPseudoLabels, SequentialMultiModalKitchens, load_rgb_frames, video_to_tensor, load_flow_frames
from pytorch_i3d import InceptionI3d
import videotransforms
import numpy as np
import pickle
import argparse
import logging
import math
import sys
import random


class EpicKitchensI3D:
    def __init__(
            self,
            num_epochs,
            init_lr,
            batch_size,
            src_domain_id,
            pseudo_type,
            use_pseudo_segs=False
    ):
        self.num_epochs = num_epochs
        self.src_domain_id = src_domain_id
        self.lr = init_lr
        self.batch_size = batch_size
        self.pseudo_type = pseudo_type
        self.use_pseudo_segs = use_pseudo_segs
        
        # Flow model
        self.flow_model = InceptionI3d(8, in_channels=2)
        self.flow_model.cuda()
        self.flow_model = nn.DataParallel(self.flow_model)
        self.flow_model.load_state_dict(torch.load(f'./trained_models/flow_mm_D1_train_random_seg_100_epochs.pt'))
        
        # RGB model
        self.rgb_model = InceptionI3d(8, in_channels=3)
        self.rgb_model.cuda()
        self.rgb_model = nn.DataParallel(self.rgb_model)
        self.rgb_model.load_state_dict(torch.load(f'./trained_models/rgb_mm_D1_train_random_seg_100_epochs.pt'))
        
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


    # To do -> Absolutely horrible, split this up and remove repetitions
    def extract_max_methods_pseudo_labels(
            self,
            prev_trg_label_path,
            new_trg_label_path
    ):
        self.rgb_model.eval()
        self.flow_model.eval()
        trg_label_df = load_pickle_data(prev_trg_label_path)
        trg_label_df["pseudo_label"] = np.nan
        trg_label_df["confidence"] = np.nan
        trg_label_df["rgb_seg_start"] = np.nan
        trg_label_df["flow_seg_start"] = np.nan
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
            for (labels, rgb_seq_img_inputs, flow_seq_img_inputs, seg_starts, too_big) in dataloader:
                logging.info(f"RGB: {len(rgb_seq_img_inputs)}, Flow: {len(flow_seq_img_inputs)}")
                # Flow
                step = 6
                flow_instance_seq_logits = torch.tensor([])
                counter = 0
                for j in range(0, len(flow_seq_img_inputs), step):
                    if j + step > len(flow_seq_img_inputs):
                        if too_big:
                            flow_seq_window_imgs = []
                            for frame_paths in flow_seq_img_inputs[j:]:
                                flow_frame_paths = [[i[0][0], i[1][0]] for i in frame_paths]
                                flow_imgs = load_flow_frames(flow_frame_paths)
                                flow_imgs = self.train_transforms(flow_imgs)
                                flow_imgs = video_to_tensor(flow_imgs)
                                flow_seq_window_imgs.append(flow_imgs)
                            temp_inputs = [torch.unsqueeze(torch.from_numpy(arr), dim=0) for arr in flow_seq_window_imgs]
                            temp_inputs = torch.tensor(torch.cat(temp_inputs)).float().cuda()
                        else:
                            temp_inputs = torch.tensor(torch.cat(flow_seq_img_inputs[j:])).float().cuda()
                    else:
                        if too_big:
                            flow_seq_window_imgs = []
                            for frame_paths in flow_seq_img_inputs[j:j+step]:
                                flow_frame_paths = [[i[0][0], i[1][0]] for i in frame_paths]
                                flow_imgs = load_flow_frames(flow_frame_paths)
                                flow_imgs = self.train_transforms(flow_imgs)
                                flow_imgs = video_to_tensor(flow_imgs)
                                flow_seq_window_imgs.append(flow_imgs)
                            temp_inputs = [torch.unsqueeze(torch.from_numpy(arr), dim=0) for arr in flow_seq_window_imgs]
                            temp_inputs = torch.tensor(torch.cat(temp_inputs)).float().cuda()
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
                        if too_big:
                            rgb_seq_window_imgs = []
                            for frame_paths in rgb_seq_img_inputs[j:]:
                                rgb_frame_paths = [i[0] for i in frame_paths]
                                rgb_imgs = load_rgb_frames(rgb_frame_paths)
                                rgb_imgs = self.train_transforms(rgb_imgs)
                                rgb_imgs = video_to_tensor(rgb_imgs)
                                rgb_seq_window_imgs.append(rgb_imgs)
                            temp_inputs = [torch.unsqueeze(torch.from_numpy(arr), dim=0) for arr in rgb_seq_window_imgs]
                            temp_inputs = torch.tensor(torch.cat(temp_inputs)).float().cuda()
                        else:
                            temp_inputs = torch.tensor(torch.cat(rgb_seq_img_inputs[j:])).float().cuda()
                    else:
                        if too_big:
                            rgb_seq_window_imgs = []
                            for frame_paths in rgb_seq_img_inputs[j:j+step]:
                                rgb_frame_paths = [i[0] for i in frame_paths]
                                rgb_imgs = load_rgb_frames(rgb_frame_paths)
                                rgb_imgs = self.train_transforms(rgb_imgs)
                                rgb_imgs = video_to_tensor(rgb_imgs)
                                rgb_seq_window_imgs.append(rgb_imgs)
                            temp_inputs = [torch.unsqueeze(torch.from_numpy(arr), dim=0) for arr in rgb_seq_window_imgs]
                            temp_inputs = torch.tensor(torch.cat(temp_inputs)).float().cuda()
                        else:
                            temp_inputs = torch.tensor(torch.cat(rgb_seq_img_inputs[j:j+step])).float().cuda()
                    rgb_logits = self.rgb_model(temp_inputs)
                    rgb_logits_reshaped = torch.reshape(rgb_logits, (rgb_logits.size(0), rgb_logits.size(1))).detach().cpu()
                    rgb_instance_seq_logits = torch.cat((rgb_instance_seq_logits, rgb_logits_reshaped))
                
                # Fuse logits for designated pseudo labelling method
                if self.pseudo_type == "max_seg":
                    pseudo_label, confidence, rgb_seg_start, flow_seg_start = self.extract_max_modal_windows(rgb_instance_seq_logits, flow_instance_seq_logits, seg_starts)
                elif self.pseudo_type == "max_agree":
                    pseudo_label, confidence, rgb_seg_start, flow_seg_start = self.extract_max_agree_modal_windows(rgb_instance_seq_logits, flow_instance_seq_logits, seg_starts)
                current_uid = int(labels[0].split("_")[1])
                trg_label_df.loc[trg_label_df["uid"] == current_uid, "pseudo_label"] = pseudo_label
                trg_label_df.loc[trg_label_df["uid"] == current_uid, "confidence"] = confidence
                trg_label_df.loc[trg_label_df["uid"] == current_uid, "rgb_seg_start"] = rgb_seg_start
                trg_label_df.loc[trg_label_df["uid"] == current_uid, "flow_seg_start"] = flow_seg_start
        trg_label_df.to_pickle(new_trg_label_path)
        logging.info(f"Extracted Pseudo Labels saved at {new_trg_label_path}")


    def extract_max_modal_windows(self, rgb_logits, flow_logits, seg_starts):
        rgb_pred, flow_pred = F.softmax(rgb_logits, dim=1), F.softmax(flow_logits, dim=1)
        rgb_max_val_at, flow_max_val_at = torch.argmax(rgb_pred), torch.argmax(flow_pred)
        rgb_seg = math.floor(rgb_max_val_at / rgb_logits.size(1))
        flow_seg = math.floor(flow_max_val_at / flow_logits.size(1))
        fused_seg_logits = rgb_logits[rgb_seg] + flow_logits[flow_seg]
        pseudo_label = torch.argmax(F.softmax(fused_seg_logits, dim=0))
        confidence = self.ce_loss(fused_seg_logits, pseudo_label)
        return pseudo_label.item(), confidence.item(), seg_starts[rgb_seg][0].item(), seg_starts[flow_seg][0].item()


    def extract_max_agree_modal_windows(self, rgb_logits, flow_logits, seg_starts):
        rgb_pred, flow_pred = F.softmax(rgb_logits, dim=1), F.softmax(flow_logits, dim=1)
        rgb_seg_pseudos, flow_seg_pseudos = torch.argmax(rgb_pred, dim=1), torch.argmax(flow_pred, dim=1)
        # RGB
        rgb_unique_labels, rgb_counts = rgb_seg_pseudos.unique(return_counts=True)
        rgb_max_agree_pseudo = rgb_unique_labels[torch.argmax(rgb_counts, dim=0).item()].item()
        max_agree_indices = torch.squeeze(torch.nonzero(rgb_seg_pseudos == rgb_max_agree_pseudo), dim=1)
        rgb_max_agree_logits, rgb_max_agree_preds, rgb_max_agree_starts = rgb_logits[max_agree_indices], rgb_pred[max_agree_indices], [seg_starts[i] for i in max_agree_indices.numpy()]
        rgb_max_val_at = torch.argmax(rgb_max_agree_preds)
        rgb_seg = math.floor(rgb_max_val_at / rgb_max_agree_logits.size(1))
        rgb_chosen_logit = rgb_max_agree_logits[rgb_seg]
        # Flow
        flow_unique_labels, flow_counts = flow_seg_pseudos.unique(return_counts=True)
        flow_max_agree_pseudo = flow_unique_labels[torch.argmax(flow_counts, dim=0).item()].item()
        max_agree_indices = torch.squeeze(torch.nonzero(flow_seg_pseudos == flow_max_agree_pseudo), dim=1)
        flow_max_agree_logits, flow_max_agree_preds, flow_max_agree_starts = flow_logits[max_agree_indices], flow_pred[max_agree_indices], [seg_starts[i] for i in max_agree_indices.numpy()]
        flow_max_val_at = torch.argmax(flow_max_agree_preds)
        flow_seg = math.floor(flow_max_val_at / flow_max_agree_logits.size(1))
        flow_chosen_logit = flow_max_agree_logits[flow_seg]
        # Fused
        fused_seg_logits = rgb_chosen_logit + flow_chosen_logit
        pseudo_label = torch.argmax(F.softmax(fused_seg_logits, dim=0))
        confidence = self.ce_loss(fused_seg_logits, pseudo_label)
        return pseudo_label.item(), confidence.item(), rgb_max_agree_starts[rgb_seg][0].item(), flow_max_agree_starts[flow_seg][0].item()


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

            # Load max method pseudo labels
            logging.info(f"Epoch {epoch}")
            trg_pseudo_labels_path = f"./label_lookup/target_pseudos_{self.pseudo_type}_epoch_{epoch}_sample_{int(trg_pseudo_sample_rate*100)}_v1.pkl"
            logging.info(f"Extracting {self.pseudo_type} pseudo labels...")
            self.extract_max_methods_pseudo_labels(prev_trg_pseudo_labels_path, trg_pseudo_labels_path)
            
            # Set model to training state
            self.rgb_model.train()
            self.flow_model.train()
            prev_trg_pseudo_labels_path = trg_pseudo_labels_path
            logging.info(f"Using labels from {trg_pseudo_labels_path}...")

            # Initialise training data with new, updated pseudo labels
            train_dataset = EpicMultiModalSrcFreeWithPseudoLabels(
                trg_pseudo_labels_path,
                trg_pseudo_sample_rate,
                transforms=self.train_transforms,
                use_pseudo_segs=self.use_pseudo_segs
            )
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            self.rgb_optim.zero_grad()
            self.flow_optim.zero_grad()
            logging.info("Beginning training...")
            sum_loss = 0.0
            num_steps_update = 6
            num_batches = len(train_dataloader)
            on_final_batch = False
            for batch_idx, (train_labels, rgb_inputs, flow_inputs, narration_ids) in enumerate(train_dataloader):
                counter += 1
                if not on_final_batch:
                    if (counter == 1 and (num_batches - batch_idx) < num_steps_update):
                        on_final_batch = True
                        update_div = num_batches - batch_idx
                    else:
                        update_div = num_steps_update

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
            logging.info(f"Epoch {epoch} training finished")
            logging.info("Evaluating test data")
            self.test_accuracy("./label_lookup/D2_test.pkl")
            logging.info("---------------------------------")
            torch.save(self.flow_model.state_dict(), f"./trained_models/flow_{self.pseudo_type}_src_free_epoch_{epoch}_sample_{int(trg_pseudo_sample_rate*100)}_v1.pt")
            torch.save(self.rgb_model.state_dict(), f"./trained_models/rgb_{self.pseudo_type}_src_free_epoch_{epoch}_sample_{int(trg_pseudo_sample_rate*100)}_v1.pt")


def load_pickle_data(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
        return df


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="Epic Kitchens Feature Extraction")
    parser.add_argument("--src_domain_id", action="store", dest="src_domain_id", default="D1")
    parser.add_argument("--trg_domain_id", action="store", dest="trg_domain_id", default="D2")
    parser.add_argument("--epochs", action="store", dest="epochs", default="5")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="8")
    parser.add_argument("--pseudo_sample_rate", action="store", dest="pseudo_sample_rate", default="0.2")
    parser.add_argument("--pseudo_type", action="store", dest="pseudo_type", default="max_seg")
    parser.add_argument("--use_pseudo_segs", action="store_true", dest="use_pseudo_segs")
    parser.set_defaults(use_pseudo_segs=True)
    args = parser.parse_args()
    logging_filename = f"train_{args.trg_domain_id}_max_seg_same_pseudo_segments_v1.log"
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
        src_domain_id=args.src_domain_id,
        pseudo_type=args.pseudo_type,
        use_pseudo_segs=args.use_pseudo_segs
    )
    logging.info(f"Use Pseudo Segments = {args.use_pseudo_segs}")
    logging.info(f"Pseudo labelling method = {args.pseudo_type}")
    logging.info("Evaluating test data on Source Only Model")
    model.test_accuracy("./label_lookup/D2_test.pkl")
    model.train_with_pseudos(
        f"./label_lookup/{args.trg_domain_id}_train.pkl",
        float(args.pseudo_sample_rate)
    )
