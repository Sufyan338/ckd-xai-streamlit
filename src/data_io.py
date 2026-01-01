import pandas as pd

def load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]  # avoid hidden whitespace bugs
    return df
