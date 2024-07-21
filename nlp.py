import polars as pl
from transformers import pipeline, BertTokenizer, BertModel
import torch

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


