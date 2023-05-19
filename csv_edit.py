import pandas as pd

# 读取所有内容
df1 = pd.read_csv('chat_80.csv')
df2 = pd.read_csv('diffusiondb.csv')

# df1 = df1.rename(columns={'image_name': 'filepath', 'prompt': 'prompt'})

# df2 = pd.read_csv('train_24w.csv')
# print(df1.shape)
# df2 = df1[df1['filepath'].str.startswith("chatgpt-images")]
# df3 = df1[df1['filepath'].str.startswith("plus80k2")]
df4 = pd.concat([df1, df2], axis=0)
print(df4.shape)
df4.to_csv("data-30w.csv", index=False)


# print(df2.shape)

# last_col = df3.pop(df3.columns[-1])
# df3.insert(0, last_col.name, last_col)
#
# def square(x):
#     return x.replace('/kaggle/input/gustavosta-stable-diffusion-prompts-sd2-v2', 'plus80k2')
#
# def square1(x):
#     return "chatgpt-images/" + x
# df1['filepath'] = df1['filepath'].apply(square1)
# df1 = pd.read_csv('plus1.csv')
# df2 = pd.read_csv('plus2.csv')
#
# df1.insert(0, 'filepath', "plus1/"+df1.index.astype(str)+".png")
# df2.insert(0, 'filepath', "plus2/"+df2.index.astype(str)+".png")
#
# df4 = pd.concat([df,df1,df2], axis=0)
# # data2 = pd.read_csv('diffusiondb2.csv')
# # data3 = pd.read_csv('diffusiondb3.csv')
# # data4 = pd.read_csv('diffusiondb4.csv')
#
# def square1(x):
#     return x.replace('/kaggle/input/diffusiondb-filtered-0-40000/0-40000/', '')
#
# def square2(x):
#     return x.replace('/kaggle/input/diffusiondb-filtered-40001-80000/40001-80000/', '')
#
# def square3(x):
#     return x.replace('/kaggle/input/diffusiondb-filtered-80001-120000/80001-120000/', '')
#
# def square4(x):
#     return x.replace('/kaggle/input/diffusiondb-filtered-120001-154320/120001-154320/', '')
# print(data1.shape)
# data1['filepath'] = data1['filepath'].apply(square1)
# data1['filepath'] = data1['filepath'].apply(square2)
# data1['filepath'] = data1['filepath'].apply(square3)
# data1['filepath'] = data1['filepath'].apply(square4)
# 生成新的文件
# df3.to_csv("zip/train_all.csv", index=False)
# df5= pd.read_csv('train_new.csv')
# print(df5.shape,df.shape,df1.shape,df2.shape)
# df = pd.read_csv('zip/train_all.csv')
# last_col = df.pop(df.columns[-1])
# df.insert(0, last_col.name, last_col)
# df3 = df3.rename(columns={'Prompt': 'prompt', 'image_path': 'filepath'})

