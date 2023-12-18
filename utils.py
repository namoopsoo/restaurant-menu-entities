




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

