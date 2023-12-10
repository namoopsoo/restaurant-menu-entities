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


DATA_DIR = "."
HF_TOKEN = os.getenv("HF_TOKEN")
LIVE_APP = os.getenv("LIVE_APP", "no")

st.title("Look at this restaurant data from Kaggle🪿")

menusdf = pd.read_csv(Path(DATA_DIR) / "restaurant-menus.csv")
menusdf = menusdf[menusdf["description"].notnull()].copy()
menusdf["concat"] = menusdf.apply(lambda x: f'{x["category"]} {x["name"]} {x["description"]}', axis=1)

restaurantsdf = pd.read_csv(Path(DATA_DIR) / "restaurants.csv")
restaurant_vec = restaurantsdf.to_dict(orient="records")
restaurant_map = {x["id"]: {"name": x["name"], "full_address": x["full_address"]} for x in restaurant_vec}

st.write(menusdf.head())
st.write("mmkay")


model_name = "all-MiniLM-L12-v2"
embedder = SentenceTransformer(
    model_name,
    use_auth_token=HF_TOKEN,
)
# embedder = SentenceTransformer(model_name)

sampledf = menusdf.sample(n=1000).reset_index()
restaurant_id_map = {i: x for i, x in enumerate(sampledf["restaurant_id"].tolist())}
# list(restaurant_id_map.items())[:5]
sentences_1000 = sampledf["concat"].tolist()

########## ########## ########## ########## ########## ##########

sentences = menusdf["concat"].tolist()
corpus = sentences_1000
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

def do_search(query):
    # st.session_state.
    print("DEBUG using, query", query, )
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = 10
    top_results = torch.topk(cos_scores, k=top_k)

    st.write(f"\nTop {top_k} most similar sentences in corpus:")

    out_vec = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = int(idx)
        restaurant_id = restaurant_id_map[idx]
        out_vec.append(
            {"text": corpus[idx].strip(), "score": "(Score: {:.4f})".format(score),
             "restaurant": restaurant_map[restaurant_id]["name"],
             "address": restaurant_map[restaurant_id]["full_address"],
             }
        )

    st.table(pd.DataFrame.from_records(out_vec))


#    if "how_many_executions" in st.session_state:
#        st.session_state["how_many_executions"] += 1
#    else:
#        st.session_state["how_many_executions"] = 1
#    st.write("Ok done with execution number ", st.session_state["how_many_executions"])





########## ########## ########## ########## ########## ##########
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

########## ########## ########## ########## ########## ##########
st.title("Paraphrase mining")
st.title("Lets run paraphrase mining on a 1000 row sample of this menu data")
st.write("First five sentences,")
st.write(sentences_1000[:5])

paraphrases = util.paraphrase_mining(embedder, sentences_1000)


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


########## ########## ########## ########## ########## ##########
st.title("Use cosine simularity for a phrase")
# st.session_state.query
query = st.text_area("Input to search for.")
button_ok = st.button("Search", on_click=do_search, args=(query, ))

st.write("Ok bye.")

