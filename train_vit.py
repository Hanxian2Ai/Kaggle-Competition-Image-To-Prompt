import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
from scipy import spatial
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import timm
from timm.utils import AverageMeter
from sentence_transformers import SentenceTransformer
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class CFG:
    model_name = 'vit_large_patch16_384'
    input_size = 384
    batch_size = 64
    num_epochs = 3
    lr = 1e-4
    seed = 42


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(CFG.seed)


class DiffusionDataset(Dataset):
    def __init__(self, df, profix, transform):
        self.df = df
        self.profix = profix
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.profix+row['filepath'])
        image = self.transform(image)
        prompt = row['prompt']
        return image, prompt


class DiffusionCollator:
    def __init__(self):
        self.st_model = SentenceTransformer(
            'sentence-transformers-222/all-MiniLM-L6-v2',
            device='cpu'
        )

    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        return images, prompt_embeddings


def get_dataloaders(
        trn_df,
        val_df,
        input_size,
        batch_size
):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trn_dataset = DiffusionDataset(trn_df, "train/", transform)
    val_dataset = DiffusionDataset(val_df, "gf_prompt/images/", transform)
    collator = DiffusionCollator()

    dataloaders = {}
    dataloaders['train'] = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=12,
        drop_last=True,
        collate_fn=collator
    )
    dataloaders['val'] = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=12,
        drop_last=False,
        collate_fn=collator
    )
    return dataloaders

def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred)
        for y_true, y_pred in zip(y_trues, y_preds)
    ])


def train(
        trn_df,
        val_df,
        model_name,
        input_size,
        batch_size,
        num_epochs,
        lr
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders = get_dataloaders(
        trn_df,
        val_df,
        input_size,
        batch_size
    )

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=384
    )
    model.set_grad_checkpointing()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    ttl_iters = num_epochs * len(dataloaders['train'])
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)
    criterion = nn.CosineEmbeddingLoss()

    best_score = -1.0

    for epoch in range(num_epochs):
        train_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.train()
        count = 0
        for X, y in dataloaders['train']:

            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            X_out = model(X)
            target = torch.ones(X.size(0)).to(device)
            loss = criterion(X_out, y, target)
            loss.backward()

            optimizer.step()
            scheduler.step()

            trn_loss = loss.item()
            trn_cos = cosine_similarity(
                X_out.detach().cpu().numpy(),
                y.detach().cpu().numpy()
            )
            train_meters['loss'].update(trn_loss, n=X.size(0))
            train_meters['cos'].update(trn_cos, n=X.size(0))
            count = count + 1
            if(count % 16 == 0):
                print('{} batch {:d} / trn/loss={:.4f}, trn/cos={:.4f}'.format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    count,
                    train_meters['loss'].avg,
                    train_meters['cos'].avg))

        print('{} Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch + 1,
            train_meters['loss'].avg,
            train_meters['cos'].avg))

        val_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.eval()
        for X, y in dataloaders['val']:
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                X_out = model(X)
                target = torch.ones(X.size(0)).to(device)
                loss = criterion(X_out, y, target)

                val_loss = loss.item()
                val_cos = cosine_similarity(
                    X_out.detach().cpu().numpy(),
                    y.detach().cpu().numpy()
                )

            val_meters['loss'].update(val_loss, n=X.size(0))
            val_meters['cos'].update(val_cos, n=X.size(0))

        print('Epoch {:d} / val/loss={:.4f}, val/cos={:.4f}'.format(
            epoch + 1,
            val_meters['loss'].avg,
            val_meters['cos'].avg))

        if val_meters['cos'].avg > best_score:
            best_score = val_meters['cos'].avg
            torch.save(model.state_dict(), f'{best_score}{model_name}.pth')


trn_df = pd.read_csv('train_38w_clean.csv')
val_df = pd.read_csv('example/prompts.csv')
# trn_df, val_df = train_test_split(trn_df, test_size=0.1, random_state=CFG.seed)
train(trn_df, val_df, CFG.model_name, CFG.input_size, CFG.batch_size, CFG.num_epochs, CFG.lr)