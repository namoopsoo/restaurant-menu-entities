import streamlit as st
import pandas as pd
import pandas as pd
import os
from pathlib import Path
from dotenv import dotenv_values
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import semantic_search, cos_sim
import torch


x = st.slider('Select a value')
st.write(x, 'squared is', x * x)

DATA_DIR = "."
HF_TOKEN = os.getenv("HF_TOKEN")

st.title("Look at this restaurant data from Kaggle🪿")

menusdf = pd.read_csv(Path(DATA_DIR) / "restaurant-menus.csv")
menusdf = menusdf[menusdf["description"].notnull()].copy()

restaurantsdf = pd.read_csv(Path(DATA_DIR) / "restaurants.csv")

st.write(menusdf.head())

st.title("Let's look at the distribution of word counts for these columns")

fig, axes = plt.subplots(figsize=(12,6), nrows=3, ncols=1)
fig.patch.set_facecolor("xkcd:mint green")
plt.tight_layout()
for i, col in enumerate(["category", "name", "description"]):

    menusdf[col + "_num_tokens"] = menusdf[col].map(lambda x: len(x.split(" "))) #  if isinstance(x, str) else 0
    ax = axes[i] #fig.add_subplot(int(f"31{i + 1}"))
    
    ax.hist(menusdf[col + "_num_tokens"], bins=50)
    ax.set(title=f"{col} num tokens")

st.pyplot(fig)

st.title("Lets run paraphrase mining on a 1000 row sample of this menu data")

model_name = "all-MiniLM-L12-v2"
embedder = SentenceTransformer(
    model_name,
    use_auth_token=HF_TOKEN,
)

menusdf["concat"] = menusdf.apply(lambda x: f'{x["category"]} {x["name"]} {x["description"]}', axis=1)
sampledf = menusdf.sample(n=1000).reset_index()
restaurant_id_map = {i: x for i, x in enumerate(sampledf["restaurant_id"].tolist())}
list(restaurant_id_map.items())[:5]
sentences_1000 = sampledf["concat"].tolist()
# sentences = np.random.choice(all_sentences, size=1000, replace=False)
# Choose 1000 first try, 
# paraphrases = util.paraphrase_mining(model, sentences)
st.write("First five sentences,")
st.write(sentences_1000[:5])

embedder = SentenceTransformer(model_name)
paraphrases = util.paraphrase_mining(embedder, sentences_1000)

st.title("Paraphrase mining")

# paraphrases_with_restaurant_ids = [x for x in paraphrases ]
for paraphrase in [row for row in paraphrases 
                   if (row[0] < .99
                   and (restaurant_id_map[row[1]] != restaurant_id_map[row[2]])) 
                  ][:5]:
    score, i, j = paraphrase
    restaurant_id_1, restaurant_id_2 = (restaurant_id_map[i], restaurant_id_map[j])
    st.write(
        f"{sentences_1000[i]} (restaurant={restaurant_id_1})\n{sentences_1000[j]} (restaurant={restaurant_id_2})\n Score: {score:.4f}\n\n" 
         )

##########
st.title("Use cosine simularity for a phrase")

query = st.text_area("Input to search for.")
top_k = st.number_input("How many top results?")

sentences = menusdf["concat"].tolist()
corpus = sentences_1000
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

query_embedding = embedder.encode(query, convert_to_tensor=True)

# We use cosine-similarity and torch.topk to find the highest 5 scores
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

st.write("\nTop 5 most similar sentences in corpus:")

out_vec = []
for score, idx in zip(top_results[0], top_results[1]):
    out_vec.append(
        {"text": corpus[idx].strip(), "score": "(Score: {:.4f})".format(score)}
    )

st.table(pd.DataFrame.from_records(out_vec))


