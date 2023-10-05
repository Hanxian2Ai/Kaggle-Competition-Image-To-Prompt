from datetime import datetime
from pathlib import Path
import time
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from sklearn.model_selection import train_test_split


clip_processor = AutoProcessor.from_pretrained("clip-vit-224")
BATCHSIZE = 64
SAVE_OPT_CKP = False
SAVE_MODEL_CKP = True
UNFREEZE_START = 20  # set it to lower number when significantly more samples are included. default 18 -> 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

run_name = f'clip-laion-l24'


def cosine_similarity_loss(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    output = -cos(pred, target).mean()
    return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        clip = AutoModel.from_pretrained("clip-vit-224")
        self.vision = clip.vision_model
        self.fc = nn.Linear(1024, 384)
        nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, x):
        x = self.vision(x)['pooler_output']
        x = self.fc(x)
        return x

def load_pretrained_model():
    model = Net()

    trainable_model_weights = False
    for name, child in model.named_children():
        if name == 'vision':
            for pn, p in child.named_parameters():
                if str(UNFREEZE_START) in pn:
                    """start unfreezing layer , the weights are trainable"""
                    trainable_model_weights = True
                p.requires_grad = trainable_model_weights
                if p.requires_grad:
                    print(f"{pn} is set to be trainable.")

    state_dict = torch.load("weight/0 clip-laion-l24 0.604210376739502.pth")
    weight_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('_orig_mod.', '') if '_orig_mod' in k else k
        weight_dict[new_k] = v
    model.load_state_dict(weight_dict)

    return model.to(device)


class IMGDataset:
    def __init__(self, df, profix, clip_processor=clip_processor):
        self.df = df
        self.input_processor = clip_processor
        self.profix = profix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.profix + row['filepath'])
        image = self.input_processor(images=image, return_tensors='pt')
        image = image.pixel_values
        image = torch.squeeze(image)
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


if __name__ == "__main__":
    """main training"""
    Path(f"../{run_name}").mkdir(exist_ok=True)

    NEPOCH = 8
    BestEpoch = 0
    BestSim = 0

    df = pd.read_csv('train_38w_clean.csv')
    df = df[:100000]
    # trn_df = pd.read_csv('train_csv/train1683554316.0060203.csv')
    # val_df = pd.read_csv('train_csv/eval1683554316.0060203.csv')
    time = str(time.time())
    # trn_df, val_df = train_test_split(df, test_size=0.1)

    # trn_df.to_csv("train_csv/train"+time+".csv", index=False)
    # val_df.to_csv("train_csv/eval"+time+".csv", index=False)
    print(f"test size: {df.shape}, train size: {df.shape}")
    nn_model = load_pretrained_model()
    nn_model = torch.compile(nn_model)
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, nn_model.parameters()), lr=4e-5, fused=True,weight_decay=0.05)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, nn_model.parameters()), lr=1e-4)

    optimizer.zero_grad()
    train_dataloader = DataLoader(dataset=IMGDataset(df, "train/"),
                                  batch_size=BATCHSIZE, shuffle=True, num_workers=8, collate_fn=DiffusionCollator())
    test_dataloader = DataLoader(dataset=IMGDataset(pd.read_csv('../example/prompts.csv'), 'train/'),
                                 batch_size=BATCHSIZE, shuffle=False, num_workers=8, collate_fn=DiffusionCollator())

    test_7_dataloader = DataLoader(dataset=IMGDataset(pd.read_csv('../example/prompts.csv'), 'train/'),
                                   batch_size=7, shuffle=False, num_workers=8, collate_fn=DiffusionCollator())
    ttl_iters = NEPOCH * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)
    iter = 0
    for epoch in range(NEPOCH):
        epoch_loss = 0
        nn_model.train()
        for batch_images, batch_targets in tqdm(train_dataloader):
            batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
            pred = nn_model(batch_images)
            cosine_loss = cosine_similarity_loss(pred, batch_targets)
            loss = cosine_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            epoch_loss += -cosine_loss.item()
            iter = iter + 1
            if (iter % 16 == 0):
                print('{} iter {:d} / trn/loss={:.4f}'.format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    iter,
                    cosine_loss))
            if epoch >= 1 and iter % 160 == 0:
                nn_model.eval()
                batch_loss = 0
                best_so = 0
                with torch.no_grad():
                    for batch_images, batch_targets in test_7_dataloader:
                        batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
                        pred = nn_model(batch_images)
                        loss = -cosine_similarity_loss(pred, batch_targets)
                        batch_loss += loss.item()
                    batch_loss /= len(test_7_dataloader)
                print(f"batch: {iter}, test_7 loss: {batch_loss}")
                if SAVE_MODEL_CKP:
                    torch.save(nn_model.state_dict(), f"{iter} {run_name} {batch_loss}.pth")
                nn_model.train()
        epoch_loss /= len(train_dataloader)
        print(f"epoch: {epoch}, training loss: {epoch_loss}")

        epoch_loss = 0
        """test7 loss"""
        with torch.no_grad():
            nn_model.eval()

            for batch_images, batch_targets in tqdm(test_7_dataloader):
                batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
                pred = nn_model(batch_images)
                print(pred)
                print(batch_targets)
                loss = -cosine_similarity_loss(pred, batch_targets)
                epoch_loss += loss.item()
            epoch_loss /= len(test_7_dataloader)
            nn_model.train()
        print(f"epoch: {epoch}, test_7 loss: {epoch_loss}")

        """test loss"""
        epoch_loss = 0
        with torch.no_grad():
            nn_model.eval()
            for batch_images, batch_targets in tqdm(test_dataloader):
                batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
                pred = nn_model(batch_images)
                loss = -cosine_similarity_loss(pred, batch_targets)
                epoch_loss += loss.item()
            epoch_loss /= len(test_dataloader)
            nn_model.train()
        print(f"epoch: {epoch}, test loss: {epoch_loss}")


        if epoch >= 0:
            oldSim = BestSim
            BestSim = epoch_loss
            BestEpoch = epoch + 1
            # print(f"save best model at{oldSim}->{BestSim} with epoch {BestEpoch}")
            if SAVE_MODEL_CKP:
                torch.save(nn_model.state_dict(), f"{epoch} {run_name} {epoch_loss}.pth")
            if SAVE_OPT_CKP:
                torch.save(optimizer.state_dict(), f"{run_name}_opt.pth")

        if epoch - 3 > BestEpoch:
            print(f"early stop at {epoch + 1} with best epoch {BestEpoch} and test similarity {BestSim}.")
            break
