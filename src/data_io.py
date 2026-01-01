import pandas as pd

def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)
