import sys
import re
import faiss
import torch
import numpy as np
import polars as pl
from pathlib import Path
import torch.nn.functional as F
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("sentence-transformers-222/sentence-transformers")
from sentence_transformers import SentenceTransformer

def check_string(string: str) -> bool:
    # Checks if the given string contains any character other than alphanumeric characters, comma, dot, hyphen or whitespace
    return bool(re.search(r'[^A-Za-z0-9,.\\-\\s]', string))

# Load data from a Parquet file
# For the purpose of illustration, the amount of data will be reduced
# pldf = pl.read_csv("diffusiondb.csv")
pldf = pl.read_parquet("metadata.parquet", columns=['image_name', 'prompt', 'width', 'height'])
#
# # Select only those images whose width and height fall between 256 and 768 pixels
pldf = pldf.filter(pl.col("width").is_between(256, 768) & pl.col("height").is_between(256, 768))
#
# Select only those prompts that have five or more words
pldf = pldf.filter(pl.col("prompt").str.split(" ").apply(lambda x: len(x)>=7))
#
# # Select only those prompts that are not blank, NULL, null, or NaN
pldf = pldf.filter(~pl.col("prompt").str.contains('^(?:\s*|NULL|null|NaN)$'))
#
#
pldf = pldf.filter(pl.col("prompt").apply(check_string))
pldf.glimpse()

#For the purpose of illustration, we will reduce the amount of data
# pldf = pldf[:10000]

model = SentenceTransformer("sentence-transformers-222/all-MiniLM-L6-v2")
vector = model.encode(pldf["prompt"].to_numpy(), batch_size=1024, show_progress_bar=True, device="cuda:1", convert_to_tensor=True)

threshold = 0.85  # Set the threshold for similarity.
n_neighbors = 5000  # Set the number of neighbors to consider.

# Perform batch processing because processing all data at once may cause resource shortage.
batch_size = 5000  # Set the batch size (i.e., the number of data items to be processed at once).
similar_vectors = []  # Create an empty list to store similar vectors.

# Create an IndexFlatIP index using the Faiss library
# The term 'IP' represents the Inner Product,
# which is equivalent to cosine similarity as it involves taking the dot product of normalized vectors.
index = faiss.IndexFlatIP(384)

# Normalize the input vector and add it to the IndexFlatIP
index.add(F.normalize(vector).cpu().numpy())

iter = 0

for i in tqdm(range(0, len(vector), batch_size)):
    # Get the target batch for processing.
    batch_data = vector.cpu().numpy()[i:i + batch_size]
    # Neighborhood search based on cosine similarity.
    similarities, indices = index.search(batch_data, n_neighbors)
    iter = iter + 1
    print(iter)
    # Extract indexes and similarities of data to be deleted.
    for j in range(similarities.shape[0]):
        close_vectors = indices[j, similarities[j] >= threshold]
        index_base = i
        # Get only the similar vectors that exclude itself
        close_vectors = close_vectors[close_vectors != index_base + j]
        similar_vectors.append((index_base + j, close_vectors))



pldf = pldf.with_columns(pl.Series(values=list(range(len(pldf))), name="index"))
pldf = pldf.filter(~pl.col("index").is_in(np.unique(np.concatenate([x for _, x in similar_vectors])).tolist()))

pldf.select(pl.col("image_name", "prompt")).write_csv("diffusiondb.csv")
print(pldf.shape)
pldf.select(pl.col("image_name", "prompt")).head()