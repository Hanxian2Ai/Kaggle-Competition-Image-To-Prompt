from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoProcessor

clip_processor = AutoProcessor.from_pretrained("/kaggle/input/clip-vit-l-14-laion2b-s32b-b82k")
BATCHSIZE = 128
SAVE_OPT_CKP = True
SAVE_MODEL_CKP = True
UNFREEZE_START = 18  # set it to lower number when significantly more samples are included.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
model_path1 = '/kaggle/input/clip-weight/clip224-l18.pth'
model_path2 = '/kaggle/input/laion-weight/2 clip-laion-l18 0.6403084424825815.pth'
input_size = 224
batch_size = 64


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        clip = AutoModel.from_pretrained("/kaggle/input/clip-vit-l-14-laion2b-s32b-b82k")
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
        model_path1,
        model_path2,
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

    state_dict = torch.load(model_path1, map_location='cuda:0')
    weight_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('_orig_mod.', '') if '_orig_mod' in k else k
        weight_dict[new_k] = v
    model.load_state_dict(weight_dict)
    model.to(device)
    model.eval()

    model2 = Net()
    state_dict2 = torch.load(model_path2, map_location='cuda:0')
    weight_dict2 = {}
    for k, v in state_dict2.items():
        new_k = k.replace('_orig_mod.', '') if '_orig_mod' in k else k
        weight_dict2[new_k] = v
    model2.load_state_dict(weight_dict2)
    model2.to(device)
    model2.eval()

    tta_preds = None
    for _ in range(2):
        preds = []
        preds2 = []
        for X in tqdm(dataloader, leave=False):
            X = X.to(device)

            with torch.no_grad():
                X_out = model(X)
                X_out2 = model2(X)
                preds.append(X_out.cpu().numpy())
                preds2.append(X_out2.cpu().numpy())

        if tta_preds is None:
            tta_preds = np.vstack(preds).flatten()
            tta_preds2 = np.vstack(preds2).flatten()
        else:
            tta_preds += np.vstack(preds).flatten()
            tta_preds2 += np.vstack(preds2).flatten()
    pred = tta_preds / 2
    pred2 = tta_preds2 / 2
    return pred * 0.5 + pred2 * 0.5


images = list(Path('/kaggle/input/stable-diffusion-image-to-prompts/images').glob('*.png'))
imgIds = [i.stem for i in images]
EMBEDDING_LENGTH = 384
imgId_eId = [
    '_'.join(map(str, i)) for i in zip(
        np.repeat(imgIds, EMBEDDING_LENGTH),
        np.tile(range(EMBEDDING_LENGTH), len(imgIds)))]

prompt_embeddings = predict(images, model_path1, model_path2, batch_size)
submission = pd.DataFrame(
    index=imgId_eId,
    data=prompt_embeddings,
    columns=['val']
).rename_axis('imgId_eId')
submission.to_csv('submission.csv')
