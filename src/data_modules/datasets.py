import pandas as pd
from torch.utils.data import Dataset



class TextClassificationDataset(Dataset):
    def __init__(self,df_path:str, text_column_name:str,label_column_name:str) -> None:
        super().__init__()
        self.df = pd.read_parquet(df_path)
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

    def __getitem__(self, idx:int) -> tuple[str,int]:
        row = self.df.iloc[str]
        text = row[self.text_column_name]
        label = row[self.label_column_name]
        return(text,label)

    def __len__(self)-> int:
        return len(self.df)


