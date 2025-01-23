import pandas as pd

# Load and inspect the dataset files
df1 = pd.read_csv("Community Questions Refined.csv")
df2 = pd.read_csv("Jenkins Docs QA.csv")
df3 = pd.read_csv("/home/kuro2806/fun/gsoc/Enhancing-LLM-with-Jenkins-Knowledge/datasets/QueryResultsUpdated.csv")

print("Community Questions Refined.csv columns:", df1.columns)
print("Jenkins Docs QA.csv columns:", df2.columns)
print("QueryResultsUpdated.csv columns:", df3.columns)
