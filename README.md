## Bronze Medal Solution For Kaggle Competition : Stable Diffusion - Image to Prompts

个人成绩：**Top2% 银牌牌** 

**赛题介绍：**

​	文本到图像模型的流行是全新的提示工程领域。一部分是艺术，一部分是悬而未决的科学，机器学习从业者和研究人员正在迅速努力理解提示与其生成的图像之间的关系。

​	将“4k”添加到提示中是使其更具摄影性的最佳方式吗？提示中的小扰动会导致高度不同的图像吗？提示关键字的顺序如何影响生成的场景？本次比赛的任务是创建一个模型，该模型可以可靠地反转生成给定图像的扩散过程。

​	为了以稳健的方式计算提示相似度，这意味着尽管字符级别存在差异。本次比赛希望创建“高质量、专注、复杂、详细、具有不真实的稳健交叉验证风格”的模型。

**比赛任务：**

1、本次比赛的目标是扭转生成文本到图像模型的典型方向：不是从文本提示生成图像，而是可以创建一个模型来预测给定生成图像的文本提示。参赛选手需要对包含由 Stable Diffusion 2.0 生成的各种（提示、图像）对的数据集进行预测，以了解潜在关系的可逆性。

2、推断生成高度详细、清晰的焦点、插图、宏伟、史诗般的 3d 渲染图像的prompt

##### 解决方案：
![](https://github.com/Hanxian2Ai/image2prompt/blob/main/md_image/Snipaste_2023-05-19_21-57-49.png)

- 数据收集、生成和清理（38w）

  - DiffusionDB 200 万图像提示子集数据集：https://poloclub.github.io/diffusiondb/
  - 文本提示符进行清洗：去重、长度筛选、字符筛选、语义相似度筛选
  - 删除重复的文本及其图片：计算相似度 使用向量搜索库[**faiss-gpu**](https://github.com/facebookresearch/faiss)
```python
model = SentenceTransformer("sentence-transformers-222/all-MiniLM-L6-v2")
vector = model.encode(pldf["prompt"].to_numpy(), batch_size=1024, show_progress_bar=True, device="cuda", convert_to_tensor=True)

threshold = 0.85  
n_neighbors = 5000  
batch_size = 5000 
similar_vectors = []  

index = faiss.IndexFlatIP(384)

index.add(F.normalize(vector).cpu().numpy())

for i in tqdm(range(0, len(vector), batch_size)):
    batch_data = vector.cpu().numpy()[i:i + batch_size]
    similarities, indices = index.search(batch_data, n_neighbors)
    for j in range(similarities.shape[0]):
        close_vectors = indices[j, similarities[j] >= threshold]
        index_base = i
        close_vectors = close_vectors[close_vectors != index_base + j]
        similar_vectors.append((index_base + j, close_vectors))
```
- model

  - 选择CLIP模型作为基准模型，包括clip-vit-large-224和clip-vit-large-336
  - 加一层全连接层，输出（1，384）维的embedding

  ```python
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
  ```

  - 第一名解决方案，在最后的输出层前加一层大的全连接层


  ```python
    ebd_dim = 1024
    fc_dim = 16 * 1024
    self.head = nn.Sequential(
      nn.Linear(ebd_dim, fc_dim),
      nn.BatchNorm1d(fc_dim),
      nn.ReLU(),
      nn.Linear(fc_dim, 384),
    )
  ```

- 数据增加

  - Mixgen([Mixgen: A new multi-modal data augmentation，WACV 2023](https://openaccess.thecvf.com/content/WACV2023W/Pretrain/html/Hao_MixGen_A_New_Multi-Modal_Data_Augmentation_WACVW_2023_paper.html))
  ![](https://github.com/Hanxian2Ai/image2prompt/blob/main/md_image/Snipaste_2023-05-21_19-59-15.png)
  - RandomHorizontalFlip(0.5)

- 训练策略

  - [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution，ICLR (Oral) 2022](https://arxiv.org/abs/2202.10054) 
  先只训练最后一层线性连接层，然后再整个模型进行微调
  ![](https://github.com/Hanxian2Ai/image2prompt/blob/main/md_image/Snipaste_2023-05-21_20-00-28.png)

- 推理策略

  - TTA

    ```python
    def predict(images,
        model_path,
        model_name,
        input_size,
        batch_size
    ):   
        tta_preds = None
        for _ in range(2):
            preds = []
            for X in tqdm(dataloader, leave=False):
                X = X.to(device)
    
                with torch.no_grad():
                    X_out = model(X).cpu().numpy()
                    # L2 normalize -- Start
                    X_out = X_out / ( np.abs(X_out).max(axis=-1, keepdims=True) + 1e-8)  
                    X_out = normalize( X_out )
                    # L2 normalize -- End
                    preds.append(X_out)
                    
            if tta_preds is None:
                tta_preds = np.vstack(preds).flatten()
            else:
                tta_preds += np.vstack(preds).flatten()
        
        return tta_preds / 2
    ```

    

  - 模型融合

    ```python
    class WeightedAverage(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.weight = nn.Linear(n, 1, bias=False)
            with torch.no_grad():
                self.weight.weight[:] = 1.0
    
        def forward(self, x, y):
            y_hat = self.weight(x)[..., 0]
            y_hat = torch.nn.functional.normalize(y_hat, dim=-1)
            cos_sim = (y * y_hat).sum(dim=-1)
            loss = -cos_sim.mean()
            return loss
    ```

    



