import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
from pathlib import Path
import traceback # For detailed error logging
from tqdm import tqdm # For scanning progress

# Helper function from training script (if not already available)
def read_flo_file(filename):
    try:
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), np.float32, count=1)
            if not np.isclose(magic[0], 202021.25):
                 return None
            width = np.frombuffer(f.read(4), np.int32, count=1)[0]
            height = np.frombuffer(f.read(4), np.int32, count=1)[0]
            if width <= 0 or height <= 0 or width * height > 15000 * 15000:
                return None
            data = np.frombuffer(f.read(), np.float32, count=-1)
            expected_elements = height * width * 2
            if data.size != expected_elements:
                 if abs(data.size - expected_elements) > 10:
                     return None
                 if data.size < expected_elements: return None
                 data = data[:expected_elements]

            return data.reshape((height, width, 2))
    except FileNotFoundError:
        return None
    except Exception as e: print(f"Error reading {filename}: {e}"); traceback.print_exc(); return None


class VideoFrameFlowDatasetNested(Dataset):
    """
    Dataset for loading consecutive video frames and their corresponding PRE-COMPUTED
    optical flow from nested directories.
    FLOW IS RETURNED AT ITS ORIGINAL RESOLUTION from the .flo file.
    Resizing/scaling should happen in the training loop.
    """
    def __init__(self, frame_base_dir, flow_base_dir, frame_prefix="im", frame_suffix=".png",
                 transform=None): # Removed flow_transform argument
        """
        Args:
            frame_base_dir (str): Path to the base directory for frames.
            flow_base_dir (str): Path to the base directory for flow files.
            frame_prefix (str): Prefix of frame image files.
            frame_suffix (str): Suffix of frame image files.
            transform (callable, optional): Transform applied to image samples (e.g., ToTensor, Resize).
        """
        self.frame_base_path = Path(frame_base_dir)
        self.flow_base_path = Path(flow_base_dir)
        self.frame_prefix = frame_prefix
        self.frame_suffix = frame_suffix
        self.transform = transform if transform else transforms.ToTensor()

        self.pairs = []
        print(f"Scanning frames in nested structure under: {self.frame_base_path}")
        print(f"Looking for corresponding flows under: {self.flow_base_path}")

        all_frames = list(self.frame_base_path.rglob(f"{self.frame_prefix}*{self.frame_suffix}"))
        print(f"Found {len(all_frames)} potential frame files. Creating pairs...")

        frames_by_dir = {}
        for f_path in all_frames:
            dir_path = f_path.parent
            if dir_path not in frames_by_dir: frames_by_dir[dir_path] = []
            frames_by_dir[dir_path].append(f_path)

        skipped_no_pair = 0
        skipped_no_flow = 0
        for dir_path, frame_list in tqdm(frames_by_dir.items(), desc="Scanning Directories"):
            try:
                sorted_frames = sorted(frame_list, key=lambda p: int(p.stem[len(self.frame_prefix):]))
            except (ValueError, IndexError): continue

            for i in range(len(sorted_frames) - 1):
                frame1_path = sorted_frames[i]
                frame2_path = sorted_frames[i+1]
                try:
                    num1 = int(frame1_path.stem[len(self.frame_prefix):])
                    num2 = int(frame2_path.stem[len(self.frame_prefix):])
                    if num2 != num1 + 1: skipped_no_pair += 1; continue
                except (ValueError, IndexError): skipped_no_pair += 1; continue

                relative_path_from_base = frame1_path.relative_to(self.frame_base_path)
                flow_path = self.flow_base_path / relative_path_from_base.with_suffix(".flo")

                if flow_path.is_file():
                    self.pairs.append((str(frame1_path), str(frame2_path), str(flow_path)))
                else:
                    skipped_no_flow += 1

        print(f"Finished scanning. Created {len(self.pairs)} valid frame/flow pairs.")
        if skipped_no_pair > 0: print(f"Skipped {skipped_no_pair} potential pairs due to non-consecutive frame numbering.")
        if skipped_no_flow > 0: print(f"Skipped {skipped_no_flow} potential pairs due to missing flow files.")
        if not self.pairs: print("WARNING: No valid pairs found.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frame1_path, frame2_path, flow_path = self.pairs[idx]

        try:
            frame1 = Image.open(frame1_path).convert('RGB')
            frame2 = Image.open(frame2_path).convert('RGB')
            flow = read_flo_file(flow_path) # Reads as HxWx2 numpy array

            if flow is None:
                raise IOError(f"Failed to load or parse flow file: {flow_path}")

            # Convert flow numpy array (HxWx2) to tensor (2xHxW), ensure float32
            # Flow tensor remains at its ORIGINAL resolution here
            flow_tensor = torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1)

            # Apply transforms ONLY to frames
            if self.transform:
                frame1 = self.transform(frame1)
                frame2 = self.transform(frame2)

            # *** NO Flow Transform or Shape Check Here ***

            return frame1, frame2, flow_tensor # Return flow at original size

        except Exception as e:
            print(f"\nERROR loading data for index {idx}: {e}")
            print(f"Paths: Frame1={frame1_path}, Frame2={frame2_path}, Flow={flow_path}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load data at index {idx}") from e