import torch
from torch.utils.data import Dataset, DataLoader
from random import randint
import pickle
import numpy as np
from PIL import Image

# New dataset should take a label list for source (pkl pandas file) and for target (pkls pandas file with additional pseudos column)
# It should then merge the source and target data into one so it can be queried alongside each other
class EpicKitchenWithPseudoLabels(Dataset):
    def __init__(self, src_labels_path, trg_pseudo_labels_path, trg_pseudo_sample_rate, is_flow=False, transforms=None):
        src_labels_df = load_pickle_data(src_labels_path)
        trg_labels_df = load_pickle_data(trg_pseudo_labels_path)
        self.transforms = transforms
        self.labels = []
        self.narration_ids = []
        self.is_flow = is_flow
        self.input_names = []

        # Load source frames and labels
        for idx, row in src_labels_df.iterrows():
            seg_img_names, start_frame = sample_train_segment(
                16,
                row["start_frame"],
                row["stop_frame"],
                row["participant_id"],
                row["video_id"],
                is_flow=is_flow
            )
            self.labels.append(row["verb_class"])
            self.narration_ids.append(f"{row['video_id']}_{row['uid']}")
            self.input_names.append(seg_img_names)
        # Load target frames and labels
        for i in range(8):
            filtered_df = trg_labels_df[trg_labels_df["pseudo_label"] == i]
            filtered_df = filtered_df.sort_values("confidence", ascending=True)
            sample_row_num = int(len(filtered_df.index)*trg_pseudo_sample_rate)
            filtered_df = filtered_df.head(sample_row_num)
            for idx, row in filtered_df.iterrows():
                seg_img_names, start_frame = sample_train_segment(
                    16,
                    row["start_frame"],
                    row["stop_frame"],
                    row["participant_id"],
                    row["video_id"],
                    is_flow=is_flow
                )
                self.labels.append(int(row["pseudo_label"]))
                self.narration_ids.append(f"{row['video_id']}_{row['uid']}")
                self.input_names.append(seg_img_names)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.is_flow:
            imgs = load_flow_frames(self.input_names[index])
            imgs = self.transforms(imgs)
            imgs = video_to_tensor(imgs)
        else:
            imgs = load_rgb_frames(self.input_names[index])
            imgs = self.transforms(imgs)
            imgs = video_to_tensor(imgs)
        return self.labels[index], imgs, self.narration_ids[index]


class EpicKitchensDataset(Dataset):
    def __init__(self, labels_path, is_flow=False, transforms=None):
        labels_df = load_pickle_data(labels_path)
        self.transforms = transforms
        self.labels = []
        self.narration_ids = []
        self.is_flow = is_flow
        self.input_names = []
        for index, row in labels_df.iterrows():
            seg_img_names, start_frame = sample_train_segment(
                16,
                row["start_frame"],
                row["stop_frame"],
                row["participant_id"],
                row["video_id"],
                is_flow=is_flow
            )
            self.labels.append(row["verb_class"])
            self.narration_ids.append(f"{row['video_id']}_{row['uid']}")
            self.input_names.append(seg_img_names)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.is_flow:
            imgs = load_flow_frames(self.input_names[index])
            imgs = self.transforms(imgs)
            imgs = video_to_tensor(imgs)
        else:
            imgs = load_rgb_frames(self.input_names[index])
            imgs = self.transforms(imgs)
            imgs = video_to_tensor(imgs)
        return self.labels[index], imgs, self.narration_ids[index]


class EpicMultiModalDataset(Dataset):
    def __init__(self, labels_path, transforms=None, use_prior_windows=False):
        labels_df = load_pickle_data(labels_path)
        self.transforms = transforms
        self.labels = []
        self.narration_ids = []
        self.video_frame_details = []
        self.use_prior_windows = use_prior_windows
        for index, row in labels_df.iterrows():
            self.labels.append(row["verb_class"])
            self.narration_ids.append(f"{row['video_id']}_{row['uid']}")
            self.video_frame_details.append({
                "start_frame": row["start_frame"],
                "stop_frame": row["stop_frame"],
                "participant_id": row["participant_id"],
                "video_id": row["video_id"],
                "rgb_seg_start": int(row["rgb_seg_start"]),
                "flow_seg_start": int(row["flow_seg_start"])
            })

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Flow
        flow_seg_img_names, flow_seg_start = sample_train_segment(
            16,
            self.video_frame_details[index]["start_frame"],
            self.video_frame_details[index]["stop_frame"],
            self.video_frame_details[index]["participant_id"],
            self.video_frame_details[index]["video_id"],
            is_flow=True,
            method="preset" if self.use_prior_windows else "random",
            preset_seg_start=self.video_frame_details[index]["flow_seg_start"] if self.use_prior_windows else None
        )
        flow_imgs = load_flow_frames(flow_seg_img_names)
        flow_imgs = self.transforms(flow_imgs)
        flow_imgs = video_to_tensor(flow_imgs)
        #RGB
        rgb_seg_img_names, rgb_seg_start = sample_train_segment(
            16,
            self.video_frame_details[index]["start_frame"],
            self.video_frame_details[index]["stop_frame"],
            self.video_frame_details[index]["participant_id"],
            self.video_frame_details[index]["video_id"],
            is_flow=False,
            method="preset" if self.use_prior_windows else "random",
            preset_seg_start=self.video_frame_details[index]["rgb_seg_start"] if self.use_prior_windows else None
        )
        rgb_imgs = load_rgb_frames(rgb_seg_img_names)
        rgb_imgs = self.transforms(rgb_imgs)
        rgb_imgs = video_to_tensor(rgb_imgs)
        return self.labels[index], rgb_imgs, flow_imgs, self.narration_ids[index], rgb_seg_start, flow_seg_start


class EpicMultiModalTestDataset(Dataset):
    def __init__(self, labels_path, transforms=None):
        labels_df = load_pickle_data(labels_path)
        self.transforms = transforms
        self.labels = []
        self.narration_ids = []
        self.input_names = []
        for index, row in labels_df.iterrows():
            rgb_seg_img_names = sample_test_segments(
                16,
                row["start_frame"],
                row["stop_frame"],
                row["participant_id"],
                row["video_id"],
                is_flow=False
            )
            flow_seg_img_names = sample_test_segments(
                16,
                row["start_frame"],
                row["stop_frame"],
                row["participant_id"],
                row["video_id"],
                is_flow=True
            )
            self.labels.append(row["verb_class"])
            self.narration_ids.append(f"{row['video_id']}_{row['uid']}")
            self.input_names.append({
                "Flow": flow_seg_img_names,
                "RGB": rgb_seg_img_names
            })

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Flow
        flow_equidist_segs = []
        for seg_img_names in self.input_names[index]["Flow"]:
            flow_seg_imgs = load_flow_frames(seg_img_names)
            flow_seg_imgs = self.transforms(flow_seg_imgs)
            flow_seg_imgs = video_to_tensor(flow_seg_imgs)
            flow_equidist_segs.append(flow_seg_imgs)
        #RGB
        rgb_equidist_segs = []
        for seg_img_names in self.input_names[index]["RGB"]:
            rgb_seg_imgs = load_rgb_frames(seg_img_names)
            rgb_seg_imgs = self.transforms(rgb_seg_imgs)
            rgb_seg_imgs = video_to_tensor(rgb_seg_imgs)
            rgb_equidist_segs.append(rgb_seg_imgs)
        return self.labels[index], rgb_equidist_segs, flow_equidist_segs, self.narration_ids[index]


class EpicMultiModalSrcFreeWithPseudoLabels(Dataset):
    def __init__(self, trg_pseudo_labels_path, trg_pseudo_sample_rate, transforms=None, use_pseudo_segs=False):
        trg_labels_df = load_pickle_data(trg_pseudo_labels_path)
        self.transforms = transforms
        self.labels = []
        self.narration_ids = []
        self.input_names = []
        self.expand_rotation = False
        self.use_pseudo_segs = use_pseudo_segs

        # Load target frames and labels
        for i in range(8):
            filtered_df = trg_labels_df[trg_labels_df["pseudo_label"] == i]
            filtered_df = filtered_df.sort_values("confidence", ascending=True)
            sample_row_num = int(len(filtered_df.index)*trg_pseudo_sample_rate)
            filtered_df = filtered_df.head(sample_row_num)
            for idx, row in filtered_df.iterrows():
                rgb_seg_img_names, _ = sample_train_segment(
                    16,
                    row["start_frame"],
                    row["stop_frame"],
                    row["participant_id"],
                    row["video_id"],
                    is_flow=False,
                    method="preset" if self.use_pseudo_segs else "random",
                    preset_seg_start=int(row["rgb_seg_start"]) if self.use_pseudo_segs else None
                )
                flow_seg_img_names, _ = sample_train_segment(
                    16,
                    row["start_frame"],
                    row["stop_frame"],
                    row["participant_id"],
                    row["video_id"],
                    is_flow=True,
                    method="preset" if self.use_pseudo_segs else "random",
                    preset_seg_start=int(row["flow_seg_start"]) if self.use_pseudo_segs else None
                )
                self.labels.append(int(row["pseudo_label"]))
                self.narration_ids.append(f"{row['video_id']}_{row['uid']}")
                self.input_names.append({
                    "Flow": flow_seg_img_names,
                    "RGB": rgb_seg_img_names
                })
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #Flow
        flow_imgs = load_flow_frames(self.input_names[index]["Flow"])
        flow_imgs = self.transforms(flow_imgs)
        flow_imgs = video_to_tensor(flow_imgs)
        #RGB
        rgb_imgs = load_rgb_frames(self.input_names[index]["RGB"])
        rgb_imgs = self.transforms(rgb_imgs)
        rgb_imgs = video_to_tensor(rgb_imgs)
        return self.labels[index], rgb_imgs, flow_imgs, self.narration_ids[index]



class EpicMultiModalWithPseudoLabels(Dataset):
    def __init__(self, src_labels_path, trg_pseudo_labels_path, trg_pseudo_sample_rate, transforms=None):
        src_labels_df = load_pickle_data(src_labels_path)
        trg_labels_df = load_pickle_data(trg_pseudo_labels_path)
        self.transforms = transforms
        self.labels = []
        self.narration_ids = []
        self.input_names = []

        # Load source frames and labels
        for idx, row in src_labels_df.iterrows():
            rgb_seg_img_names, rgb_start_frame = sample_train_segment(
                16,
                row["start_frame"],
                row["stop_frame"],
                row["participant_id"],
                row["video_id"],
                is_flow=False
            )
            flow_seg_img_names, flow_start_frame = sample_train_segment(
                16,
                row["start_frame"],
                row["stop_frame"],
                row["participant_id"],
                row["video_id"],
                is_flow=True
            )
            self.labels.append(row["verb_class"])
            self.narration_ids.append(f"{row['video_id']}_{row['uid']}")
            self.input_names.append({
                "Flow": flow_seg_img_names,
                "RGB": rgb_seg_img_names
            })
        # Load target frames and labels
        for i in range(8):
            filtered_df = trg_labels_df[trg_labels_df["pseudo_label"] == i]
            filtered_df = filtered_df.sort_values("confidence", ascending=True)
            sample_row_num = int(len(filtered_df.index)*trg_pseudo_sample_rate)
            filtered_df = filtered_df.head(sample_row_num)
            for idx, row in filtered_df.iterrows():
                rgb_seg_img_names, rgb_start_frame = sample_train_segment(
                    16,
                    row["start_frame"],
                    row["stop_frame"],
                    row["participant_id"],
                    row["video_id"],
                    is_flow=False
                )
                flow_seg_img_names, flow_start_frame = sample_train_segment(
                    16,
                    row["start_frame"],
                    row["stop_frame"],
                    row["participant_id"],
                    row["video_id"],
                    is_flow=True
                )
                self.labels.append(int(row["pseudo_label"]))
                self.narration_ids.append(f"{row['video_id']}_{row['uid']}")
                self.input_names.append({
                    "Flow": flow_seg_img_names,
                    "RGB": rgb_seg_img_names
                })
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #Flow
        flow_imgs = load_flow_frames(self.input_names[index]["Flow"])
        flow_imgs = self.transforms(flow_imgs)
        flow_imgs = video_to_tensor(flow_imgs)
        #RGB
        rgb_imgs = load_rgb_frames(self.input_names[index]["RGB"])
        rgb_imgs = self.transforms(rgb_imgs)
        rgb_imgs = video_to_tensor(rgb_imgs)
        return self.labels[index], rgb_imgs, flow_imgs, self.narration_ids[index]


class SequentialMultiModalKitchens(Dataset):
    def __init__(self, labels_path, class_num, temporal_window=16, step=2, sequential_overlap=4, transforms=None):
        labels_df = load_pickle_data(labels_path)
        filtered_df = labels_df[labels_df["verb_class"] == class_num]
        self.labels = []
        self.input_names = []
        self.seq_window_start_ratios = []
        self.sequential_overlap = sequential_overlap
        self.transforms = transforms
        for index, row in filtered_df.iterrows():
            clip_length = row["stop_frame"] - row["start_frame"]
            num_windows, total_window_len = get_num_seq_windows(clip_length, temporal_window, self.sequential_overlap)
            frame_start_numbers = get_start_frame_numbers(
                row["start_frame"],
                row["stop_frame"],
                total_window_len,
                num_windows,
                self.sequential_overlap
            )
            self.labels.append(f"{row['verb_class']}_{row['uid']}")
            self.input_names.append({
                "sequential": {
                    "Flow": sample_test_sequential_frames(
                        frame_start_numbers,
                        temporal_window,
                        True,
                        row["participant_id"],
                        row["video_id"]
                    ),
                    "RGB": sample_test_sequential_frames(
                        frame_start_numbers,
                        temporal_window,
                        False,
                        row["participant_id"],
                        row["video_id"]
                    )
                },
                "start_frames": frame_start_numbers
            })

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        flow_seq_window_imgs = []
        rgb_seq_window_imgs = []
        if len(self.input_names[index]["start_frames"]) >= 50:
            return self.labels[index], self.input_names[index]["sequential"]["RGB"], self.input_names[index]["sequential"]["Flow"], self.input_names[index]["start_frames"], True
        # Flow
        for frame_paths in self.input_names[index]["sequential"]["Flow"]:
            flow_imgs = load_flow_frames(frame_paths)
            flow_imgs = self.transforms(flow_imgs)
            flow_imgs = video_to_tensor(flow_imgs)
            flow_seq_window_imgs.append(flow_imgs)
        # RGB
        for frame_paths in self.input_names[index]["sequential"]["RGB"]:
            rgb_imgs = load_rgb_frames(frame_paths)
            rgb_imgs = self.transforms(rgb_imgs)
            rgb_imgs = video_to_tensor(rgb_imgs)
            rgb_seq_window_imgs.append(rgb_imgs)
        return self.labels[index], rgb_seq_window_imgs, flow_seq_window_imgs, self.input_names[index]["start_frames"], False



class SequentialClassKitchens(Dataset):
    def __init__(self, labels_path, class_num, temporal_window=16, is_flow=False, step=2, sequential_overlap=4, transforms=None):
        labels_df = load_pickle_data(labels_path)
        filtered_df = labels_df[labels_df["verb_class"] == class_num]
        self.labels = []
        self.input_names = []
        self.seq_window_start_ratios = []
        self.is_flow = is_flow
        self.sequential_overlap = sequential_overlap
        self.transforms = transforms
        for index, row in filtered_df.iterrows():
            clip_length = row["stop_frame"] - row["start_frame"]
            num_windows, total_window_len = get_num_seq_windows(clip_length, temporal_window, self.sequential_overlap)
            frame_start_numbers = get_start_frame_numbers(
                row["start_frame"],
                row["stop_frame"],
                total_window_len,
                num_windows,
                self.sequential_overlap
            )
            self.seq_window_start_ratios.append([((i - row["start_frame"])/clip_length) for i in frame_start_numbers])
            self.labels.append(f"{row['verb_class']}_{row['uid']}")
            self.input_names.append({
                "sequential": sample_test_sequential_frames(
                    frame_start_numbers,
                    temporal_window,
                    self.is_flow,
                    row["participant_id"],
                    row["video_id"]
                ),
                "frames": {
                    "start": row["start_frame"],
                    "stop": row["stop_frame"]
                }
            })

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq_window_imgs = []
        for frame_paths in self.input_names[index]["sequential"]:
            if self.is_flow:
                imgs = load_flow_frames(frame_paths)
                imgs = self.transforms(imgs)
                #central_imgs = load_flow_frames(self.input_names[index]["central"])
            else:
                imgs = load_rgb_frames(frame_paths)
                imgs = self.transforms(imgs)
                #central_imgs = load_rgb_frames(self.input_names[index]["central"])
            imgs = video_to_tensor(imgs)
            #central_imgs = video_to_tensor(central_imgs)
            seq_window_imgs.append(imgs)
        #return self.labels[index], seq_window_imgs, central_imgs, self.seq_window_start_ratios[index]
        return self.labels[index], seq_window_imgs, self.seq_window_start_ratios[index]


def sample_test_sequential_frames(start_frames, temporal_window_len, is_flow, part_id, video_id):
    seq_frame_names = []
    temporal_window_len = 32
    for start_frame in start_frames:
        if is_flow:
            seq_frame_names.append([
                [
                    f"./epic_kitchens_data/flow/{part_id}/{video_id}/u/frame_{str(int((start_frame+i)/2)).zfill(10)}.jpg",
                    f"./epic_kitchens_data/flow/{part_id}/{video_id}/v/frame_{str(int((start_frame+i)/2)).zfill(10)}.jpg"
                ] for i in range(0, temporal_window_len, 2)
            ])
        else:
            seq_frame_names.append(
                [f"./epic_kitchens_data/rgb/{part_id}/{video_id}/frame_{str((start_frame+i)).zfill(10)}.jpg" for i in range(0, temporal_window_len, 2)]
            )
    return seq_frame_names


def get_num_seq_windows(clip_length, window_len, overlap):
    window_len = 32
    num_windows = round(clip_length/window_len)
    window_total_len = (num_windows*window_len) - (overlap*(num_windows-1))
    if window_total_len > clip_length and window_len <= clip_length:
        num_windows -= 1
        window_total_len = (num_windows*window_len) - (overlap*(num_windows-1))
    return num_windows, window_total_len


def get_start_frame_numbers(start_frame, stop_frame, total_window_len, num_windows, overlap):
    centre_frame = int(start_frame + ((stop_frame - start_frame)/2))
    half_window_len = int(total_window_len / 2)
    frame_start_numbers = []
    temporal_window = 32
    for i in range(num_windows):
        frame_start_numbers.append((centre_frame-half_window_len) + (temporal_window*i) - (overlap*i))
    return frame_start_numbers


def load_pickle_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data


def video_to_tensor(frames):
    return frames.transpose([3, 0, 1, 2])


def load_flow_frames(flow_filenames):
    for i, group in enumerate(flow_filenames):
        img_u = np.array(Image.open(group[0]))
        img_v = np.array(Image.open(group[1]))
        if i == 0:
            frames = np.array([[img_u, img_v]])
        else:
            frames = np.concatenate((frames, np.array([[img_u, img_v]])), axis=0)
    frames = frames.transpose([0, 2, 3, 1])
    return (((frames / 255) * 2) - 1)


def load_rgb_frames(rgb_filenames):
    for i, filename in enumerate(rgb_filenames):
        img_matrix = np.array(Image.open(filename))
        if i == 0:
            frames = np.array([img_matrix])
        else:
            frames = np.concatenate((frames, np.array([img_matrix])), axis=0)
    return (((frames / 255) * 2) - 1)


# Function to return random segment in training clip to match MMSADA training
def sample_train_segment(
        temporal_window,
        start_frame,
        end_frame,
        part_id,
        video_id,
        is_flow,
        step=2,
        method="random",
        preset_seg_start=None
):
    half_frame = int(temporal_window/2)
    #step = 2
    seg_img_names = []
    segment_start = int(start_frame) + (step*half_frame)
    segment_end = int(end_frame) + 1 - (step*half_frame)
    # Write a comment to explain this weirdness
    if segment_start >= segment_end:
        segment_start = int(start_frame)
        segment_end = int(end_frame)
    if segment_start <= half_frame*step+1:
        segment_start = half_frame*step+2
    if method == "central":
        centre_frame = int((segment_end - segment_start)/2) + segment_start
    elif method == "preset" and preset_seg_start is not None:
        centre_frame = preset_seg_start
    else:
        centre_frame = randint(segment_start, segment_end)
    for i in range(centre_frame-(step*half_frame), centre_frame+(step*half_frame), step):
        if is_flow:
            seg_img_names.append([
                f"./epic_kitchens_data/flow/{part_id}/{video_id}/u/frame_{str(int(i/2)).zfill(10)}.jpg",
                f"./epic_kitchens_data/flow/{part_id}/{video_id}/v/frame_{str(int(i/2)).zfill(10)}.jpg"
            ])
        else:
            seg_img_names.append(f"./epic_kitchens_data/rgb/{part_id}/{video_id}/frame_{str(i).zfill(10)}.jpg")
    return seg_img_names, centre_frame


# Function to return 5 equidistant segments to match MMSADA testing regime
def sample_test_segments(temporal_window, start_frame, end_frame, part_id, video_id, is_flow, step=2, method="random"):
    half_frame=int(temporal_window/2)
    seg_img_names = []
    segment_start = int(start_frame) + (step*half_frame)
    segment_end = int(end_frame) + 1 - (step*half_frame)
    if segment_start >= segment_end:
        segment_start = int(start_frame)
        segment_end = int(end_frame)
    if segment_start <= half_frame*step+1:
        segment_start = half_frame*step+2
    for centre_frame in np.linspace(segment_start, segment_end, 7, dtype=np.int32)[1:-1]:
        seg_i = []
        for i in range(centre_frame - (step*half_frame), centre_frame + (step*half_frame), step):
            if is_flow:
                seg_i.append([
                    f"./epic_kitchens_data/flow/{part_id}/{video_id}/u/frame_{str(int(i/2)).zfill(10)}.jpg",
                    f"./epic_kitchens_data/flow/{part_id}/{video_id}/v/frame_{str(int(i/2)).zfill(10)}.jpg"
                ])
            else:
                seg_i.append(f"./epic_kitchens_data/rgb/{part_id}/{video_id}/frame_{str(i).zfill(10)}.jpg")
        seg_img_names.append(seg_i)
    return seg_img_names


if __name__ == "__main__":
    train_dataset = EpicKitchensDataset(labels_path="/user/work/rg16964/label_lookup/D2_train.pkl", is_flow=False)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    for (labels, rgb_inputs) in train_dataloader:
        print(labels)
        print(labels.shape)
        break
