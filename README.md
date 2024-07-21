---
title: restaurant menu entities
emoji: 🙂
sdk: streamlit
app_file: app.py
pinned: false
---



# how to run basic demo 
## prerequisites 
### tokens 
Refresh hugging face tokens for embeddings, into `.env` , `HF_TOKEN`
### requirements 
a new `python3 -m venv ~/.python_venvs/blahvenv` with , 
```
pip install -r requirements.txt
```
## locally 
streamlit run app.py

![](assets/Screenshot2024-07-14-14.40.06.png)


### Add this uber eats data into postgresql , 

```python

import polars as pl
menusdf = pl.read_csv("restaurant-menus.csv")
restaurantsdf = pl.read_csv("restaurants.csv")

menusdf = menusdf.with_columns(
    #price_cleaned=menusdf.select(pl.col("price").map_elements(lambda x:x.split(" ")[0])),
    price=pl.col("price").map_elements(lambda x:x.split(" ")[0]).cast(pl.Float32),
)

df = menusdf.join(
    restaurantsdf.rename({"name": "restaurant_name"}), 
    left_on="restaurant_id", 
    right_on="id")

# apply an id column.
df = df.with_columns(
    id=df.with_row_index()["index"],
)

from utils import embed_and_load_into_db, make_vectorstore_thing

vectorstore = make_vectorstore_thing()

# clean slate
vectorstore.drop_tables()

sample_df = df.sample(n=10000, seed=214)  # seed chosen to maximize the "chicken parmesan query results"

ids = utils.embed_and_load_into_db(
    sample_df.to_dicts(),
    id_col="id",
    text_col="name",
    metadata_cols=["price", "restaurant_name", "full_address", "lat", "lng"],
)
```

Quick search 

```python
from utils import search_pg_vector

vectorstore = make_vectorstore_thing()

docs_with_scores = search_pg_vector("chicken parm")
```
