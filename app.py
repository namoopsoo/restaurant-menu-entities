


import streamlit as st

import pandas as pd
import pandas as pd
from pathlib import Path
from dotenv import dotenv_values
import matplotlib.pyplot as plt


x = st.slider('Select a value')
st.write(x, 'squared is', x * x)


DATA_DIR = "."


menusdf = pd.read_csv(Path(DATA_DIR) / "restaurant-menus.csv")
menusdf = menusdf[menusdf["description"].notnull()].copy()
restaurantsdf = pd.read_csv(Path(DATA_DIR) / "restaurants.csv")

st.write("head")


print(menusdf.head())

st.write("Let's look at the distribution of word counts for these columns")


fig, axes = plt.subplots(figsize=(12,6), nrows=3, ncols=1)
fig.patch.set_facecolor("xkcd:mint green")
plt.tight_layout()
for i, col in enumerate(["category", "name", "description"]):

    menusdf[col + "_num_tokens"] = menusdf[col].map(lambda x: len(x.split(" "))) #  if isinstance(x, str) else 0
    ax = axes[i] #fig.add_subplot(int(f"31{i + 1}"))
    
    ax.hist(menusdf[col + "_num_tokens"], bins=50)
    ax.set(title=f"{col} num tokens")

st.pyplot(fig)

st.write("ok")
