import streamlit as st
import torch
import polars as pl
# import pandas as pd
import os
from pathlib import Path
from dotenv import dotenv_values, load_dotenv
# import matplotlib.pyplot as plt

from utils import search_pg_vector
from nlp import is_this_about_food, extract_geo_location_stuff

device = "cuda:0" if torch.cuda.is_available() else "cpu"

DATA_DIR = "."
LIVE_APP = os.getenv("LIVE_APP", "no")

if LIVE_APP == "no":
    load_dotenv()


st.title("Lets search uber eats data with a query")

def do_search():
    query = st.session_state.my_query
    print("DEBUG using, query", query, )

    if not query:
        st.write("Query empty, doing nothing")
        return

    #  TODO is it on topic?
    class_df, on_topic, food_pred = is_this_about_food(query)

    st.write("Is this query about food?")
    st.table(class_df)

    if not on_topic:
        st.write("Looks like this query is off topic.")
        return

    top_k = 10
    docs_scored = search_pg_vector(query, k=top_k)

    results_df = pl.from_dicts(
        [{**(row[0].metadata), "item": row[0].page_content, "score": row[1]} for row in docs_scored]
    )

    st.write(f"\nTop {top_k} most similar sentences in corpus: to \"{query}\"")

    st.table(results_df)

    location_tokens, all_other_tokens = extract_geo_location_stuff(query)
    if location_tokens:
        st.write(f"location tokens, {location_tokens}")
        st.write(f"all other tokens, {all_other_tokens}")
    else:
        st.write("no location tokens found.")




with st.form(key='my_form'):
    query = st.text_area("Input to search for.", key="my_query")
    # slider_input = st.slider('My slider', 0, 10, 5, key='my_slider')
    # checkbox_input = st.checkbox('Yes or No', key='my_checkbox')
    submit_button = st.form_submit_button(label='Search', on_click=do_search)


st.write("Ok bye.")
