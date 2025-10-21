from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import os
import cv2
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD  = [0.229, 0.224, 0.225]

# train_tf = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])
# test_tf = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])


img_size=224
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]


base_transform = alb.Compose([
    alb.Resize(img_size,img_size),
    alb.Normalize(mean=mean, std=std),
    ToTensorV2(),
])


class FFPP_Dataset(Dataset):
    def __init__(self, root_dir, phase='train', img_size=224, augment=False):
        assert phase in ['train','test']
        self.phase = phase

        self.dataset = datasets.ImageFolder(os.path.join(root_dir, phase))
        self.samples = self.dataset.samples

        print(f"📁 Loaded {len(self.samples)} images from {root_dir} "
              f"({sum(l==0 for _,l in self.samples)} fake / "
              f"{sum(l==1 for _,l in self.samples)} real)")
        
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if augment and phase == 'train':
            self.transform = alb.Compose([
                alb.Resize(img_size, img_size),
                alb.HorizontalFlip(p=0.5),
                alb.OneOf([
                    alb.MotionBlur(p=0.3),
                    alb.GaussianBlur(p=0.3),
                    alb.GaussNoise(p=0.3)
                ], p=0.5),
                alb.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
                alb.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            self.transform = alb.Compose([
                alb.Resize(img_size, img_size),
                alb.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image loading fail: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transform
        image = self.transform(image=image)["image"]
        return image, label

# train_set = datasets.ImageFolder("/home/toi/research/face_forgery_detection/datasets/FFPP_C23_baseline/train", transform=base_transform)
# test_set  = datasets.ImageFolder("/home/toi/research/face_forgery_detection/datasets/FFPP_C23_baseline/test",  transform=base_transform)

# train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=8, pin_memory=True)
# test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
train_set = FFPP_Dataset("/home/toi/research/face_forgery_detection/datasets/FFPP_C23_baseline", phase='train', img_size=224, augment=True)
test_set  = FFPP_Dataset("/home/toi/research/face_forgery_detection/datasets/FFPP_C23_baseline", phase='test', img_size=224, augment=False)

if __name__ == "__main__":
    # Test dataset loading
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    for images, labels in train_loader:
        print(f"Train batch - images: {images.shape}, labels: {labels}")
        break

    for images, labels in test_loader:
        print(f"Test batch - images: {images.shape}, labels: {labels}")
        break