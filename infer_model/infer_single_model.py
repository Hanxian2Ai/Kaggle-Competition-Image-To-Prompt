from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoProcessor

clip_processor = AutoProcessor.from_pretrained("/kaggle/input/openai-clip/")
BATCHSIZE = 128
SAVE_OPT_CKP = True
SAVE_MODEL_CKP = True
UNFREEZE_START = 18  # set it to lower number when significantly more samples are included.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
model_path = '/kaggle/input/15w-18-1e-4/2 clip224-l18 0.605536552992734.pth'
input_size = 224
batch_size = 64


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        clip = AutoModel.from_pretrained("/kaggle/input/openai-clip/")
        self.vision = clip.vision_model
        self.fc = nn.Linear(1024, 384)

    def forward(self, x):
        out = self.vision(x)['pooler_output']
        return self.fc(out)


class IMGDataset:
    def __init__(self, images, clip_processor=clip_processor):
        self.images = images
        self.input_processor = clip_processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.input_processor(images=image, return_tensors='pt')
        image = image.pixel_values
        image = torch.squeeze(image)
        return image


def predict(
        images,
        model_path,
        batch_size
):
    dataloader = DataLoader(
        dataset=IMGDataset(images, clip_processor),
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )

    model = Net()
    state_dict = torch.load(model_path,map_location='cuda:0')
    weight_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('_orig_mod.', '') if '_orig_mod' in k else k
        weight_dict[new_k] = v
    model.load_state_dict(weight_dict)
    model.to(device)
    model.eval()

    tta_preds = None
    for _ in range(2):
        preds = []
        for X in tqdm(dataloader, leave=False):
            X = X.to(device)

            with torch.no_grad():
                X_out = model(X)
                preds.append(X_out.cpu().numpy())

        if tta_preds is None:
            tta_preds = np.vstack(preds).flatten()
        else:
            tta_preds += np.vstack(preds).flatten()

    return tta_preds / 2


images = list(Path('/kaggle/input/stable-diffusion-image-to-prompts/images').glob('*.png'))
imgIds = [i.stem for i in images]
EMBEDDING_LENGTH = 384
imgId_eId = [
    '_'.join(map(str, i)) for i in zip(
        np.repeat(imgIds, EMBEDDING_LENGTH),
        np.tile(range(EMBEDDING_LENGTH), len(imgIds)))]

prompt_embeddings = predict(images, model_path, batch_size)
submission = pd.DataFrame(
    index=imgId_eId,
    data=prompt_embeddings,
    columns=['val']
).rename_axis('imgId_eId')
submission.to_csv('submission.csv')
