
import os 
import logging
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector


def search(query, corpus):
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


def embed_and_load_into_db(documents, id_col, text_col, metadata_cols):


    password = os.getenv("PG_PASSWORD")
    connection = f"postgresql+psycopg://michal:{password}@localhost:5432/langchain"  # Uses psycopg3!
    collection_name = "my_docs"
    embeddings = CohereEmbeddings()

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    
    docs = [
        Document(
            page_content=row[text_col],
            metadata={"id": row[id_col], **{k: row[k] for k in metadata_cols}},
        )
        for row in documents
    ]

    result = vectorstore.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])


    return result
