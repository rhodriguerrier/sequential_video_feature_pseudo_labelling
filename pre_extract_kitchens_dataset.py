import torch
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
import numpy as np

# Static transformer ----------------------------------

class EpicExtractedAugFt(Dataset):
    def __init__(self, domain, mode):
        self.domain = domain
        self.mode = mode
        self.data = {"rgb": {}, "flow": {}}
        if self.mode == "train":
            self.labels = torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_labels_25_equidistant_windows_aug_0.npy"))
            for i in range(0, 6):
                self.data["rgb"].update({str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{self.mode}/{self.domain}/{self.domain}_{self.mode}_rgb_25_equidistant_windows_aug_{i}.npy"))})
                self.data["flow"].update({str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{self.mode}/{self.domain}/{self.domain}_{self.mode}_flow_25_equidistant_windows_aug_{i}.npy"))})
        else:
            self.labels = torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_labels_5_equidistant_windows_aug_0.npy"))
            self.data["rgb"].update({"0": torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{self.mode}/{self.domain}/{self.domain}_{self.mode}_rgb_5_equidistant_windows_aug_0.npy"))})
            self.data["flow"].update({"0": torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{self.mode}/{self.domain}/{self.domain}_{self.mode}_flow_5_equidistant_windows_aug_0.npy"))})
        
    def __len__(self):
        return self.labels.size(0)

    def get_random_seg_nums(self):
        random_idx = []
        while len(random_idx) < 5:
            r = random.randint(0, 24)
            if r not in random_idx:
                random_idx.append(r)
        random_idx.sort()
        return random_idx

    def join_augmented_segments(self, index, random_seg_nums, modality):
        vec_ls = []
        for seg_num in random_seg_nums:
            aug_num = random.randint(0, 5)
            seg_aug = self.data[modality][str(aug_num)][index]
            vec_ls.append(torch.unsqueeze(seg_aug[seg_num], dim=0))
        return torch.cat(vec_ls)

    def __getitem__(self, index):
        if self.mode == "train":
            random_idx_rgb, random_idx_flow = self.get_random_seg_nums(), self.get_random_seg_nums()
            return self.labels[index], self.join_augmented_segments(index, random_idx_rgb, "rgb"), self.join_augmented_segments(index, random_idx_flow, "flow"), np.array(random_idx_rgb), np.array(random_idx_flow), index
        else:
            return self.labels[index], self.data["rgb"]["0"][index], self.data["flow"]["0"][index], index


class EpicExtractedAugFtWithPseudos(Dataset):
    def __init__(self, domain, mode, labels_path, sample_rate, add_one_noisy=False):
        self.mode = mode
        self.domain = domain
        self.add_one_noisy = add_one_noisy
        self.data = {"rgb": {}, "flow": {}}
        for i in range(0, 6):
            self.data["rgb"].update({str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{self.mode}/{self.domain}/{self.domain}_{self.mode}_rgb_25_equidistant_windows_aug_{i}.npy"))})
            self.data["flow"].update({str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{self.mode}/{self.domain}/{self.domain}_{self.mode}_flow_25_equidistant_windows_aug_{i}.npy"))})
        pseudo_labels_df = labels_path
        self.sampled_pseudo_labels = pd.DataFrame(columns=["label_idx", "actual_label", "pseudo_label", "confidence"])
        for i in range(8):
            filtered_df = pseudo_labels_df[pseudo_labels_df["pseudo_label"] == i]
            filtered_df = filtered_df.sort_values("confidence", ascending=True)
            sample_row_num = int(len(filtered_df.index)*sample_rate)
            filtered_df = filtered_df.head(sample_row_num)
            self.sampled_pseudo_labels = pd.concat((
                self.sampled_pseudo_labels,
                filtered_df
            ))
        self.random_noisy_idx = random.randint(0, len(self.sampled_pseudo_labels.index)-1)

    def __len__(self):
        return len(self.sampled_pseudo_labels.index)

    def join_augmented_segments(self, index, random_seg_nums, modality):
        vec_ls = []
        for seg_num in random_seg_nums:
            aug_num = random.randint(0, 5)
            seg_aug = self.data[modality][str(aug_num)][index]
            vec_ls.append(torch.unsqueeze(seg_aug[seg_num], dim=0))
        return torch.cat(vec_ls)

    def __getitem__(self, index):
        row = self.sampled_pseudo_labels.iloc[index]
        label_idx = row["label_idx"]
        rgb_seg_idx = [int(x) for x in row["rgb_segs"].split(",")]
        flow_seg_idx = [int(x) for x in row["flow_segs"].split(",")]
        if self.add_one_noisy and index == self.random_noisy_idx:
            print("In add one noisy label if")
            label_same = True
            while label_same:
                noisy_label = random.randint(0, 7)
                if noisy_label != row["pseudo_label"]:
                    print(f"Pseudo Label {row['pseudo_label']} -> Noisy Label {noisy_label}")
                    pseudo_label = noisy_label
                    label_same = False
        else:
            pseudo_label = row["pseudo_label"]
        return pseudo_label, self.join_augmented_segments(label_idx, rgb_seg_idx, "rgb"), self.join_augmented_segments(label_idx, flow_seg_idx, "flow")


class EpicExtractedFt(Dataset):
    def __init__(self, domain, mode, random_windows=True):
        self.random_windows = random_windows
        if self.random_windows:
            len_type = "25_equidistant"
        else:
            len_type = "5_equidistant"
        self.labels = torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_labels_{len_type}_windows_aug_0.npy"))
        self.rgb_ft = torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_rgb_{len_type}_windows_aug_0.npy"))
        self.flow_ft = torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_flow_{len_type}_windows_aug_0.npy"))

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        if self.random_windows:
            random_idx_rgb = []
            while len(random_idx_rgb) < 5:
                r = random.randint(0, 24)
                if r not in random_idx_rgb:
                    random_idx_rgb.append(r)
            random_idx_flow = []
            while len(random_idx_flow) < 5:
                r = random.randint(0, 24)
                if r not in random_idx_flow:
                    random_idx_flow.append(r)
            random_idx_rgb.sort()
            random_idx_flow.sort()
            return self.labels[index], self.rgb_ft[index][np.array(random_idx_rgb)], self.flow_ft[index][np.array(random_idx_rgb)], index, np.array(random_idx_rgb), np.array(random_idx_flow)
        return self.labels[index], self.rgb_ft[index], self.flow_ft[index], index


# Variable transformer ----------------------------------

class EpicVariableAugFtWithPseudos(Dataset):
    def __init__(self, domain, mode, pseudo_labels_df, sample_rate):
        self.domain = domain
        self.mode = mode
        self.data = {"rgb": {}, "flow": {}}
        for i in range(0, 6):
            self.data["rgb"].update({
                str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_rgb_all_windows_aug_{i}.npy"))
            })
            self.data["flow"].update({
                str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_flow_all_windows_aug_{i}.npy"))
            })
        self.labels = np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_labels_all_windows_aug_0.npy")
        self.labels_wo_id = np.unique(self.labels)
        self.sampled_pseudo_labels = pd.DataFrame(columns=["label_id", "actual_label", "pseudo_label", "confidence"])
        for i in range(8):
            filtered_df = pseudo_labels_df[pseudo_labels_df["pseudo_label"] == i]
            filtered_df = filtered_df.sort_values("confidence", ascending=True)
            sample_row_num = int(len(filtered_df.index)*sample_rate)
            filtered_df = filtered_df.head(sample_row_num)
            self.sampled_pseudo_labels = pd.concat((
                self.sampled_pseudo_labels,
                filtered_df
            ))

    def __len__(self):
        return len(self.sampled_pseudo_labels.index)

    def __getitem__(self, index):
        label_id = self.sampled_pseudo_labels.iloc[index]["label_id"]
        pseudo_label = self.sampled_pseudo_labels.iloc[index]["pseudo_label"]
        label_indices = np.where(self.labels == label_id)[0]
        return pseudo_label, {"index": label_id, "len": len(label_indices)}

    def get_padded_batch(self, indices, pad_len):
        padded_rgb_ls, padded_flow_ls, mask_ls = [], [], []
        for index in indices:
            label_indices = np.where(self.labels == index)[0]
            mask_ls.append(
                torch.unsqueeze(torch.cat((
                    torch.zeros(len(label_indices), dtype=torch.bool),
                    torch.ones(pad_len-len(label_indices), dtype=torch.bool),
                    torch.zeros(len(label_indices), dtype=torch.bool),
                    torch.ones(pad_len-len(label_indices), dtype=torch.bool),
                    torch.zeros(1, dtype=torch.bool)
                ), dim=0), dim=0)
            )
            if self.mode == "train":
                rgb_aug_segs, flow_aug_segs = self.get_augmented_segs(label_indices)
            else:
                rgb_aug_segs, flow_aug_segs = self.data["rgb"]["0"][label_indices], self.data["flow"]["0"][label_indices]
            if len(label_indices) < pad_len:
                padded_rgb_ls.append(
                    torch.unsqueeze(
                        torch.cat((rgb_aug_segs, torch.zeros(pad_len-len(label_indices), 1024))),
                        dim=1
                    )
                )
                padded_flow_ls.append(
                    torch.unsqueeze(
                        torch.cat((flow_aug_segs, torch.zeros(pad_len-len(label_indices), 1024))),
                        dim=1
                    )
                )
            else:
                padded_rgb_ls.append(torch.unsqueeze(rgb_aug_segs, dim=1))
                padded_flow_ls.append(torch.unsqueeze(flow_aug_segs, dim=1))
        return torch.cat(padded_rgb_ls, dim=1), torch.cat(padded_flow_ls, dim=1), torch.cat(mask_ls)

    def get_augmented_segs(self, label_indices):
        rgb_seg_ls, flow_seg_ls = [], []
        for idx in label_indices:
            aug_num = random.randint(0, 5)
            rgb_seg_ls.append(torch.unsqueeze(self.data["rgb"][str(aug_num)][idx], dim=0))
            flow_seg_ls.append(torch.unsqueeze(self.data["flow"][str(aug_num)][idx], dim=0))
        return torch.cat(rgb_seg_ls), torch.cat(flow_seg_ls)


class EpicVariableAugFt(Dataset):
    def __init__(self, domain, mode):
        self.domain = domain
        self.mode = mode
        self.labels = np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_labels_all_windows_aug_0.npy")
        self.data = {
            "rgb": {
                "0": torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_rgb_all_windows_aug_0.npy"))
            },
            "flow": {
                "0": torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_flow_all_windows_aug_0.npy"))
            }
        }
        if self.mode == "train":
            for i in range(1, 6):
                self.data["rgb"].update({
                    str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_rgb_all_windows_aug_{i}.npy"))
                })
                self.data["flow"].update({
                    str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_flow_all_windows_aug_{i}.npy"))
                })
        self.labels_wo_id = np.unique(self.labels)

    def __len__(self):
        return self.labels_wo_id.shape[0]

    def __getitem__(self, index):
        label_indices = np.where(self.labels == self.labels_wo_id[index])[0]
        return self.labels_wo_id[index], {"index": index, "len": len(label_indices)}

    def get_padded_batch(self, indices, pad_len):
        padded_rgb_ls, padded_flow_ls, mask_ls = [], [], []
        for index in indices:
            label_indices = np.where(self.labels == self.labels_wo_id[index])[0]
            mask_ls.append(
                torch.unsqueeze(torch.cat((
                    torch.zeros(len(label_indices), dtype=torch.bool),
                    torch.ones(pad_len-len(label_indices), dtype=torch.bool),
                    torch.zeros(len(label_indices), dtype=torch.bool),
                    torch.ones(pad_len-len(label_indices), dtype=torch.bool),
                    torch.zeros(1, dtype=torch.bool)
                ), dim=0), dim=0)
            )
            if self.mode == "train":
                rgb_aug_segs, flow_aug_segs = self.get_augmented_segs(label_indices)
            else:
                rgb_aug_segs, flow_aug_segs = self.data["rgb"]["0"][label_indices], self.data["flow"]["0"][label_indices]
            if len(label_indices) < pad_len:
                padded_rgb_ls.append(
                    torch.unsqueeze(
                        torch.cat((rgb_aug_segs, torch.zeros(pad_len-len(label_indices), 1024))),
                        dim=1
                    )
                )
                padded_flow_ls.append(
                    torch.unsqueeze(
                        torch.cat((flow_aug_segs, torch.zeros(pad_len-len(label_indices), 1024))),
                        dim=1
                    )
                )
            else:
                padded_rgb_ls.append(torch.unsqueeze(rgb_aug_segs, dim=1))
                padded_flow_ls.append(torch.unsqueeze(flow_aug_segs, dim=1))
        return torch.cat(padded_rgb_ls, dim=1), torch.cat(padded_flow_ls, dim=1), torch.cat(mask_ls)

    def get_augmented_segs(self, label_indices):
        rgb_seg_ls, flow_seg_ls = [], []
        for idx in label_indices:
            aug_num = random.randint(0, 5)
            rgb_seg_ls.append(torch.unsqueeze(self.data["rgb"][str(aug_num)][idx], dim=0))
            flow_seg_ls.append(torch.unsqueeze(self.data["flow"][str(aug_num)][idx], dim=0))
        return torch.cat(rgb_seg_ls), torch.cat(flow_seg_ls)


class EpicVariableFt(Dataset):
    def __init__(self, domain, mode):
        self.labels = np.load(f"./extracted_data/source_model_ft/{domain}_{mode}_labels_all_windows.npy")
        self.labels_wo_id = np.unique(self.labels)
        self.rgb_ft = torch.from_numpy(np.load(f"./extracted_data/source_model_ft/{domain}_{mode}_rgb_all_windows.npy"))
        self.flow_ft = torch.from_numpy(np.load(f"./extracted_data/source_model_ft/{domain}_{mode}_flow_all_windows.npy"))

    def __len__(self):
        return self.labels_wo_id.shape[0]

    def __getitem__(self, index):
        label_indices = np.where(self.labels == self.labels_wo_id[index])[0]
        return self.labels_wo_id[index], {"index": index, "len": len(label_indices)}

    def get_padded_batch(self, indices, pad_len):
        padded_rgb_ls, padded_flow_ls, mask_ls = [], [], []
        for index in indices:
            label_indices = np.where(self.labels == self.labels_wo_id[index])[0]
            mask_ls.append(
                torch.unsqueeze(torch.cat((
                    torch.zeros(len(label_indices), dtype=torch.bool),
                    torch.ones(pad_len-len(label_indices), dtype=torch.bool),
                    torch.zeros(len(label_indices), dtype=torch.bool),
                    torch.ones(pad_len-len(label_indices), dtype=torch.bool),
                    torch.zeros(1, dtype=torch.bool)
                ), dim=0), dim=0)
            )
            if len(label_indices) < pad_len:
                padded_rgb_ls.append(
                    torch.unsqueeze(
                        torch.cat((self.rgb_ft[label_indices], torch.zeros(pad_len-len(label_indices), 1024))),
                        dim=1
                    )
                )
                padded_flow_ls.append(
                    torch.unsqueeze(
                        torch.cat((self.flow_ft[label_indices], torch.zeros(pad_len-len(label_indices), 1024))),
                        dim=1
                    )
                )
            else:
                padded_rgb_ls.append(torch.unsqueeze(self.rgb_ft[label_indices], dim=1))
                padded_flow_ls.append(torch.unsqueeze(self.flow_ft[label_indices], dim=1))
        return torch.cat(padded_rgb_ls, dim=1), torch.cat(padded_flow_ls, dim=1), torch.cat(mask_ls)

# MLP -------------------------------------------------------------

class EpicConcatAugFtWithPseudos(Dataset):
    def __init__(self, domain, mode, pseudo_labels_df, sample_rate, add_one_noisy=False):
        self.mode = mode
        self.domain = domain
        self.add_one_noisy = add_one_noisy
        self.data = {"rgb": {}, "flow": {}}
        for i in range(0, 6):
            self.data["rgb"].update({str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_rgb_25_equidistant_windows_aug_{i}.npy"))})
            self.data["flow"].update({str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_flow_25_equidistant_windows_aug_{i}.npy"))})
        self.sampled_pseudo_labels = pd.DataFrame(columns=["label_idx", "actual_label", "pseudo_label", "confidence"])
        for i in range(8):
            filtered_df = pseudo_labels_df[pseudo_labels_df["pseudo_label"] == i]
            filtered_df = filtered_df.sort_values("confidence", ascending=True)
            sample_row_num = int(len(filtered_df.index)*sample_rate)
            filtered_df = filtered_df.head(sample_row_num)
            self.sampled_pseudo_labels = pd.concat((
                self.sampled_pseudo_labels,
                filtered_df
            ))
        self.random_noisy_idx = random.randint(0, len(self.sampled_pseudo_labels.index)-1)

    def __len__(self):
        return len(self.sampled_pseudo_labels.index)

    def join_augmented_segments(self, seg_nums, modality, index):
        seg_ls = []
        for idx in seg_nums:
            aug_num = random.randint(0, 5)
            seg_ls.append(torch.unsqueeze(self.data[modality][str(aug_num)][index][idx], dim=0))
        return torch.cat(seg_ls)

    def __getitem__(self, index):
        row = self.sampled_pseudo_labels.iloc[index]
        label_idx = row["label_idx"]
        rgb_seg_idx = [int(x) for x in row["rgb_segs"].split(",")]
        flow_seg_idx = [int(x) for x in row["flow_segs"].split(",")]
        rgb_ft = self.join_augmented_segments(rgb_seg_idx, "rgb", label_idx)
        flow_ft = self.join_augmented_segments(flow_seg_idx, "flow", label_idx)
        if self.add_one_noisy and index == self.random_noisy_idx:
            print("In add one noisy if")
            label_same = True
            while label_same:
                noisy_label = random.randint(0, 7)
                if noisy_label != row["pseudo_label"]:
                    print(f"Pseudo Label {row['pseudo_label']} -> Noisy Label {noisy_label}")
                    pseudo_label = noisy_label
                    label_same = False
        else:
            pseudo_label = row["pseudo_label"]
        return pseudo_label, torch.flatten(rgb_ft), torch.flatten(flow_ft)


class EpicConcatAugFt(Dataset):
    def __init__(self, domain, mode):
        self.domain = domain
        self.mode = mode
        self.data = {"rgb": {}, "flow": {}}
        if self.mode == "train":
            self.labels = torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_labels_25_equidistant_windows_aug_0.npy"))
            for i in range(0, 6):
                self.data["rgb"].update({str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_rgb_25_equidistant_windows_aug_{i}.npy"))})
                self.data["flow"].update({str(i): torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_flow_25_equidistant_windows_aug_{i}.npy"))})
        else:
            self.labels = torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_labels_5_equidistant_windows_aug_0.npy"))
            self.data["rgb"].update({"0": torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_rgb_5_equidistant_windows_aug_0.npy"))})
            self.data["flow"].update({"0": torch.from_numpy(np.load(f"./extracted_data/ft_with_augmentations/{mode}/{domain}/{domain}_{mode}_flow_5_equidistant_windows_aug_0.npy"))})

    def __len__(self):
        return self.labels.size(0)

    def get_random_seg_nums(self):
        random_idx = []
        while len(random_idx) < 5:
            r = random.randint(0, 24)
            if r not in random_idx:
                random_idx.append(r)
        random_idx.sort()
        return random_idx

    def join_augmented_segments(self, seg_nums, modality, index):
        seg_ls = []
        for idx in seg_nums:
            aug_num = random.randint(0, 5)
            seg_ls.append(torch.unsqueeze(self.data[modality][str(aug_num)][index][idx], dim=0))
        return torch.cat(seg_ls)

    def __getitem__(self, index):
        if self.mode == "train":
            random_idx_rgb, random_idx_flow = self.get_random_seg_nums(), self.get_random_seg_nums()
            rgb_ft = self.join_augmented_segments(random_idx_rgb, "rgb", index)
            flow_ft = self.join_augmented_segments(random_idx_flow, "flow", index)
            return self.labels[index], torch.flatten(rgb_ft), torch.flatten(flow_ft), np.array(random_idx_rgb), np.array(random_idx_flow), index
        else:
            rgb_ft = self.data["rgb"]["0"][index]
            flow_ft = self.data["flow"]["0"][index]
            return self.labels[index], torch.flatten(rgb_ft), torch.flatten(flow_ft), index


class EpicConcatFt(Dataset):
    def __init__(self, domain, mode, random_windows=True):
        self.random_windows = random_windows
        if self.random_windows:
            len_type = "25_equidistant"
        else:
            len_type = "5_equidistant"
        self.labels = torch.from_numpy(np.load(f"./extracted_data/source_model_ft/{domain}_{mode}_labels_{len_type}_windows.npy"))
        self.rgb_ft = torch.from_numpy(np.load(f"./extracted_data/source_model_ft/{domain}_{mode}_rgb_{len_type}_windows.npy"))
        self.flow_ft = torch.from_numpy(np.load(f"./extracted_data/source_model_ft/{domain}_{mode}_flow_{len_type}_windows.npy"))

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        if self.random_windows:
            random_idx_rgb = []
            while len(random_idx_rgb) < 5:
                r = random.randint(0, 24)
                if r not in random_idx_rgb:
                    random_idx_rgb.append(r)
            random_idx_flow = []
            while len(random_idx_flow) < 5:
                r = random.randint(0, 24)
                if r not in random_idx_flow:
                    random_idx_flow.append(r)
            random_idx_rgb.sort()
            random_idx_flow.sort()
            return self.labels[index], torch.flatten(self.rgb_ft[index][np.array(random_idx_rgb)]), torch.flatten(self.flow_ft[index][np.array(random_idx_rgb)]), index, random_idx_rgb, random_idx_flow
        return self.labels[index], torch.flatten(self.rgb_ft[index]), torch.flatten(self.flow_ft[index]), index


def load_pickle_data(path):
    with open(path, "rb") as f:
        return pickle.load(path)
