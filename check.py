import pandas as pd

files = ["Community Questions Refined.csv", "Jenkins Docs QA.csv", "QueryResultsUpdated.csv"]
for file in files:
    try:
        df = pd.read_csv(file, quoting=3)
        assert {'question', 'answer'}.issubset(df.columns)
        print(f"{file}: OK")
    except Exception as e:
        print(f"{file}: FAILED - {str(e)}")
