import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import torch

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import semantic_search, cos_sim
from datasets import load_dataset


DATA_DIR = "."
HF_TOKEN = os.getenv("HF_TOKEN")
LIVE_APP = os.getenv("LIVE_APP", "no")

loaded_embeddings = load_dataset("namoopsoo-org/2023-12-17-nypl-dishes-embeddings-10k-sample")
corpus_embeddings = torch.from_numpy(
    loaded_embeddings["train"].to_pandas().to_numpy()
).to(torch.float)

dishdf_sample_10k = load_dataset("namoopsoo-org/namoopsoo/2023-12-17-nypl-dishes-10k-sample")["train"].to_pandas()
corpus = dishdf_sample_10k["name"].tolist()

# Try that search again, 
# query = "chicken parmesan sandwich"
# query_embedding = model.encode(query, convert_to_tensor=True)    
# cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
# top_k = 5
# top_results = torch.topk(cos_scores, k=top_k)
# print("query:", query, [[corpus[i], score] for (score, i) in zip(top_results[0], top_results[1])], "\n\n")



model_name = "all-MiniLM-L12-v2"
embedder = SentenceTransformer(
    model_name,
    use_auth_token=HF_TOKEN,
)
HF_TOKEN

def do_search():

    query = st.session_state.my_query
    print("DEBUG using, query", query, )

    if not query:
        st.write("Query empty, doing nothing")
        return
    # st.session_state.
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = 10
    top_results = torch.topk(cos_scores, k=top_k)

    st.write(f"\nTop {top_k} most similar sentences in corpus: to \"{query}\"")

    out_vec = []
    for score, i in zip(top_results[0], top_results[1]):
        i = int(i)
        # restaurant_id = restaurant_id_map[i]
        out_vec.append(
            {"text": corpus[i].strip(), "score": "(Score: {:.4f})".format(score),
             # "restaurant": restaurant_map[restaurant_id]["name"],
             # "address": restaurant_map[restaurant_id]["full_address"],
             }
        )

    st.table(pd.DataFrame.from_records(out_vec))


st.title("Use cosine simularity for a phrase")
st.write(f"This below demo takes a phrase, embeds it with the model, \"{model_name}\", (from https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2 )  and shows the ranked cosine similarity output against a 10k-row sample from the NYPL Dish dataset available at https://www.kaggle.com/datasets/prashant111/discover-the-menu/ . The dataset embeddings are stored in a provate huggingface dataset ")
# st.session_state.query
# button_ok = st.button("Search")



with st.form(key='my_form'):
    query = st.text_area("Input to search for.", key="my_query")
    # slider_input = st.slider('My slider', 0, 10, 5, key='my_slider')
    # checkbox_input = st.checkbox('Yes or No', key='my_checkbox')
    submit_button = st.form_submit_button(label='Search', on_click=do_search)



st.write("Ok bye.")
