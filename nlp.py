import polars as pl
import torch
from transformers import pipeline, BertTokenizer, BertModel


# Load Hugging Face pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def check_if_on_topic(query, topics):
    
    classification = classifier(query, topics, multi_label=True)
    return classification


def is_this_about_food(query):
    #
    topics = [
        "this is about food",
        "food",
        "this is about a restaurant",
        "restaurants",
        "food places",
        "this is about food places",
    ]
    len_food_topics = len(topics)
    classifications = check_if_on_topic(query, topics)

    class_df = pl.from_records([classifications["labels"], classifications["scores"]], schema=["label", "score"])
    
    food_pred = sum(classifications["scores"]) / len_food_topics

    THRESHOLD = 0.65  # TODO can tune this.
    on_topic = food_pred >= THRESHOLD
    return class_df, on_topic, food_pred


def extract_geo_location_stuff(query):

    classifier = pipeline(task="ner")
    # query = "I'm on the corner of 14th stret and Broadway and I am trying to get to 59th street and Central Park West ok how can I travel?"

    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"

    tokenizer = BertTokenizer.from_pretrained(model_name)

    preds = classifier(query)
    preds = [
        {
            "entity": pred["entity"],
            "score": round(pred["score"], 4),
            "index": pred["index"],
            "word": pred["word"],
            "start": pred["start"],
            "end": pred["end"],
        }
        for pred in preds
    ]
    print("DEBUG ner extract.")
    print(*preds, sep="\n")

    tokens = tokenizer.tokenize(query)

    location_indexes = [x["index"] - 1
                        for x in preds
                        if x["entity"] in ["I-LOC"]]

    location_tokens = [tokens[i] for i, x in enumerate(tokens)
                       if i in location_indexes]

    all_other_tokens = [tokens[i] for i, x in enumerate(tokens)
                        if i not in location_indexes]

    return location_tokens, all_other_tokens
