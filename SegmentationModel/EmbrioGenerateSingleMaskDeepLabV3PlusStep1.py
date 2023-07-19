import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from tqdm import tqdm
import os


NETWORK_FOLDER_PATH = 'FOLDER_WITH_NETWORK.zip'
MODEL_PATH = os.path.join(NETWORK_FOLDER_PATH,'deepLabV3Plus_193.zip')
output_dir = os.path.join (NETWORK_FOLDER_PATH,'MasksGeneratedByTheNetwork')
INPUT_IMAGE_PATH = 'FOLDER_WITH_INPUT_FILE/INPUT_FILE_NAME.png'
cudaId = 0

DETECTION_THRESHOLD=0.5
INPUT_IMAGE_HEIGHT=512
INPUT_IMAGE_WIDTH=512


class SegmentationDatasetEmbrio(Dataset):
    def __init__(self, imageFolder, transform=None):
        self.imagePath = imageFolder
        self.transforms=transform

    def __len__(self):
        return  1

    def __getitem__(self, idx):
        index=idx
        image = cv2.imread(self.imagePath)
        if self.transforms is not None:
            image = self.transforms(image)
        return (image)


ID_TO_NAME = {
    0: "GV",
    1: "FPB",
}

def get_device(cudaId):
    if torch.cuda.is_available():
        DEVICE = f'cuda:{cudaId}'
        print('Running on the GPU')
    else:
        DEVICE = "cpu"
        print('Running on the CPU')
    return DEVICE


if(os.path.isdir(output_dir) == False):
    os.mkdir(output_dir)


transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                                 ])

test_dataset = SegmentationDatasetEmbrio(INPUT_IMAGE_PATH, transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = smp.DeepLabV3Plus(
    encoder_name="resnext101_32x8d",
    encoder_weights=None,
    in_channels=3,
    classes=2,
)

model.load_state_dict(torch.load(MODEL_PATH))

DEVICE = get_device(cudaId)

if torch.cuda.is_available():
    model.cuda(DEVICE)

with torch.no_grad():
    model.eval()
    for (i, (x)) in tqdm(enumerate(test_loader)):
        (x) = (x.to(DEVICE))
        y_pred = model(x)

        predNormalizedBool = y_pred > DETECTION_THRESHOLD
        y_pred[predNormalizedBool] = 1
        y_pred[predNormalizedBool == False] = 0

        for label in range(0, 2):
            labelId = ID_TO_NAME[label]
            maskFilePath = os.path.join(output_dir,f'_{labelId}.csv')

            np.savetxt(maskFilePath, y_pred.cpu().data.numpy()[0][label].round().astype(int),fmt='%i', delimiter=',')

print('Files generated')