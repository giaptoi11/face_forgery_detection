import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ================= CONFIG ====================

IMG_SIZE = 224
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# List of known manipulation methods in FF++
FAKE_METHODS = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

# Albumentations preprocessing (same as MoE-FFD / CFM)
base_transform = alb.Compose([
    alb.Resize(IMG_SIZE, IMG_SIZE),
    alb.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ================= UNIVERSAL DATASET ====================

class UniversalDataset(Dataset):
    """
    Universal dataset for deepfake detection.
    Supports both:
        - FaceForensics++ (train/test splits from .json)
        - Any benchmark dataset (CelebDF, DFDC_P, DFD, WDF, DF40, etc.)

    Automatically detects dataset structure based on folder names.
    Loads all frames available in 'frames/' folders (no 20-frame sampling).
    """

    def __init__(self, root, dataset_name='FFPP', split_json=None, phase='train'):
        """
        Args:
            root (str): Path to dataset root directory.
            dataset_name (str): 'FFPP', 'CDF', 'DFDC_P', etc.
            split_json (str): Path to FF++ split file (train.json / test.json).
            phase (str): 'train' or 'test'.
        """
        super().__init__()
        assert phase in ['train', 'test'], "phase must be 'train' or 'test'"
        self.root = root
        self.dataset_name = dataset_name.upper()
        self.phase = phase
        self.samples = []

        # === FaceForensics++ (requires split JSON) ===
        if self.dataset_name in ['FFPP', 'FF++']:
            assert split_json is not None, "FF++ requires a split JSON file"
            with open(split_json, 'r') as f:
                video_pairs = json.load(f)
            self.samples = self._load_ffpp(video_pairs)

        # === Other benchmark datasets (CDF, DFDC_P, DFD, etc.) ===
        elif self.dataset_name.upper() == 'CDFV1':
            print(f"Loading {self.dataset_name.upper()} dataset...")
            self.samples = self._load_celebdf()
        elif self.dataset_name.upper() == 'CDFV2':
            print(f"Loading {self.dataset_name.upper()} dataset...")
            self.samples = self._load_celebdf()
        elif self.dataset_name.upper() == 'DFDC':
            print(f"Loading {self.dataset_name.upper()} dataset...")
            self.samples = self._load_dfdc()   
        else:
            print(f"Loading {self.dataset_name.upper()} dataset...")
            self.samples = self._load_generic_dataset()

        print(f"📁 [{self.dataset_name}] Loaded {len(self.samples)} images "
              f"({sum(l==0 for _,l in self.samples)} real / {sum(l==1 for _,l in self.samples)} fake)")

    # ---------- (1) FF++ Loader ----------
    def _load_ffpp(self, video_pairs):
        """
        Loads FF++ (C23) following the official folder structure:
            manipulated_sequences/<method>/c23/frames/<src>_<tgt>_<method>/
            original_sequences/(actors|youtube)/c23/frames/<id>/
        """
        samples = []

        # === FAKE videos ===
        for pair in tqdm(video_pairs, desc=f"[FFPP-{self.phase}] Fake videos"):
            if not isinstance(pair, list):
                continue
            src, tgt = pair
            for method in FAKE_METHODS:
                fake_dir = os.path.join(
                    self.root, "manipulated_sequences", method, "c23", "frames",
                    f"{src}_{tgt}"
                )
                if not os.path.exists(fake_dir):
                    continue
                for img_name in os.listdir(fake_dir):
                    if img_name.lower().endswith((".png", ".jpg")):
                        samples.append((os.path.join(fake_dir, img_name), 1))  # label=1 (fake)

        # === REAL videos (both source and target IDs) ===
        real_ids = set()
        for pair in video_pairs:
            if isinstance(pair, list):
                real_ids.update(pair)

        for rid in tqdm(real_ids, desc=f"[FFPP-{self.phase}] Real videos"):
            for sub in ["actors", "youtube"]:
                real_dir = os.path.join(
                    self.root, "original_sequences", sub, "c23", "frames", rid
                )
                if not os.path.exists(real_dir):
                    continue
                for img_name in os.listdir(real_dir):
                    if img_name.lower().endswith((".png", ".jpg")):
                        samples.append((os.path.join(real_dir, img_name), 0))  # label=0 (real)

        return samples

    # ---------- (2) Generic Cross-Dataset Loader ----------
    def _load_generic_dataset(self):
        """
        Loads any dataset with folders named like:
            root/
              real/ or originals/
              fakes/ or swapped/ or method names/
        The label is inferred from folder names.
        """
        samples = []
        for method in os.listdir(self.root):
            method_path = os.path.join(self.root, method)
            if not os.path.isdir(method_path):
                continue

            # Determine label by folder name
            if any(x in method.lower() for x in ['real', 'original']):
                label = 0
            else:
                label = 1

            # Each subfolder = one video (contains frames/)
            for v in os.listdir(method_path):
                v_path = os.path.join(method_path, v)
                if not os.path.isdir(v_path):
                    continue

                # Handle nested structure (e.g., <video>/frames/)
                if os.path.exists(os.path.join(v_path, "frames")):
                    frame_path = os.path.join(v_path, "frames")
                else:
                    frame_path = v_path

                for img_name in os.listdir(frame_path):
                    if img_name.lower().endswith((".png", ".jpg")):
                        samples.append((os.path.join(frame_path, img_name), label))
        return samples
    
    def _load_celebdf(self):
        samples = []
        subfolders = ['Celeb-real', 'YouTube-real', 'Celeb-synthesis']

        for sub in subfolders:
            sub_path = os.path.join(self.root, sub)
            if not os.path.exists(sub_path):
                continue

            if sub in ['Celeb-real', 'YouTube-real']:
                label = 0  # real
            else:
                label = 1  # fake

            # each video under sub has frames/
            for vid in os.listdir(f"{sub_path}/frames"):
                vid_path = os.path.join(f"{sub_path}/frames", vid)
                # print(vid_path)
                if not os.path.exists(vid_path):
                    continue

                for img_name in os.listdir(vid_path):
                    if img_name.lower().endswith((".png", ".jpg")):
                        samples.append((os.path.join(vid_path, img_name), label))

        print(f"Loaded {len(samples)} frames from CelebDF "
            f"({sum(l==0 for _,l in samples)} real / {sum(l==1 for _,l in samples)} fake)")
        return samples
    

    def _load_dfdc(self):
        """
        Loads the DFDC dataset using metadata.json.
        Structure:
            DFDC/
            test/
                frames/<video_id>/
                metadata.json
        """
        samples = []
        meta_path = os.path.join(self.root, "test", "metadata.json")
        frames_root = os.path.join(self.root, "test", "frames")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.json not found at {meta_path}")

        # Load metadata file
        with open(meta_path, "r") as f:
            meta = json.load(f)

        for vid_name, info in tqdm(meta.items(), desc="[DFDC] Loading"):
            label = info.get("is_fake")
            # label = 0 if label_str == "real" else 1
            vid_id = os.path.splitext(vid_name)[0]

            vid_dir = os.path.join(frames_root, vid_id)
            if not os.path.exists(vid_dir):
                continue

            for img_name in os.listdir(vid_dir):
                if img_name.lower().endswith((".png", ".jpg")):
                    img_path = os.path.join(vid_dir, img_name)
                    samples.append((img_path, label))

        print(f"Loaded {len(samples)} frames from DFDC "
            f"({sum(l==0 for _,l in samples)} real / {sum(l==1 for _,l in samples)} fake)")
        return samples

    # ---------- PyTorch dataset interface ----------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"⚠️ Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = base_transform(image=img)["image"]
        return img, label


# ================= DATALOADER WRAPPER ====================

def get_dataloader(root, dataset_name='FFPP', split_json=None, batch_size=32, phase='train'):
    """
    Returns a PyTorch DataLoader for any deepfake dataset.

    Args:
        root (str): Dataset root path.
        dataset_name (str): 'FFPP', 'CDF', 'DFDC_P', etc.
        split_json (str): Path to train/test JSON for FF++.
        batch_size (int): Mini-batch size.
        phase (str): 'train' or 'test'.
    """
    dataset = UniversalDataset(root, dataset_name=dataset_name, split_json=split_json, phase=phase)
    shuffle = True if phase == 'train' else False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, drop_last=True)
    return loader


if __name__ == "__main__":
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

    imgs, labels = next(iter(train_loader))
    print(imgs.shape, labels.unique())
    # torch.Size([32, 3, 224, 224]) tensor([0, 1])

    #cdf_root = "datasets/CelebDF_v2"
    # cdf_loader = get_dataloader(cdf_root, dataset_name="CDF", phase="test", batch_size=32)

    # dfdc_root = "datasets/DFDC_P"
    # dfdc_loader = get_dataloader(dfdc_root, dataset_name="DFDC_P", phase="test", batch_size=32)
