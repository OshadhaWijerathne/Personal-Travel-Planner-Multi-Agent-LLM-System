# agents/data_retrieval.py
import pandas as pd

df = pd.read_csv("dataset_samples_20.csv")

def data_retrieval_agent():
    return df["reference_information"][0]

