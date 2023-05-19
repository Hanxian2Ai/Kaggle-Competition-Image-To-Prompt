## Bronze Medal Solution For 'Kaggle Complete : Stable Diffusion - Image to Prompts'

kaggle主页 :  https://www.kaggle.com/hanxian0820

个人成绩：**Top8% 铜牌**  🥉🥉🥉

**赛题介绍：**

​	文本到图像模型的流行是全新的提示工程领域。一部分是艺术，一部分是悬而未决的科学，机器学习从业者和研究人员正在迅速努力理解提示与其生成的图像之间的关系。

​	将“4k”添加到提示中是使其更具摄影性的最佳方式吗？提示中的小扰动会导致高度不同的图像吗？提示关键字的顺序如何影响生成的场景？本次比赛的任务是创建一个模型，该模型可以可靠地反转生成给定图像的扩散过程。

​	为了以稳健的方式计算提示相似度，这意味着尽管字符级别存在差异。本次比赛希望创建“高质量、专注、复杂、详细、具有不真实的稳健交叉验证风格”的模型。

**比赛任务：**

1、本次比赛的目标是扭转生成文本到图像模型的典型方向：不是从文本提示生成图像，而是可以创建一个模型来预测给定生成图像的文本提示。参赛选手需要对包含由 Stable Diffusion 2.0 生成的各种（提示、图像）对的数据集进行预测，以了解潜在关系的可逆性。

2、推断生成高度详细、清晰的焦点、插图、宏伟、史诗般的 3d 渲染图像的prompt

##### 解决方案：

- 数据收集、生成和清理（38w）
  - DiffusionDB 200 万图像提示子集数据集：https://poloclub.github.io/diffusiondb/
  - 文本提示符进行清洗：去重、长度筛选、字符筛选、语义相似度筛选
  - 删除重复的文本及其图片：计算相似度 使用向量搜索库**faiss-gpu**

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

- 数据增加
  - mixGen
  - RandomHorizontalFlip(0.5)
- 训练策略
  - 

- 推理策略
  - TTA
  - 模型融合
    - 使用随机空间搜索
