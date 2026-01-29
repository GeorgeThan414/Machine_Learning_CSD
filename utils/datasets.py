import torch
from torch.utils.data import Dataset
import pandas as pd


class HouseholdDataset(Dataset):
    """
    Train / valid:
        returns (x, y, survey_id, hhid)
    Test:
        returns (x, survey_id, hhid)
    """

    def __init__(self, features_csv, gt_csv=None):
        self.df = pd.read_csv(features_csv)

        # required identifiers
        self.survey_id = self.df["survey_id"].to_numpy()
        self.hhid = self.df["hhid"].to_numpy()

        self.has_targets = gt_csv is not None

        if self.has_targets:
            gt = pd.read_csv(gt_csv)[["survey_id", "hhid", "cons_ppp17"]]
            self.df = self.df.merge(gt, on=["survey_id", "hhid"], how="inner")

        # build X
        drop_cols = ["survey_id", "hhid"]
        if self.has_targets:
            drop_cols.append("cons_ppp17")

        self.X = torch.tensor(self.df.drop(columns=drop_cols).values,dtype=torch.float32)

        if self.has_targets:
            self.y = torch.log1p(torch.tensor(self.df["cons_ppp17"].values,dtype=torch.float32))
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.has_targets:
            return self.X[idx], self.y[idx], self.survey_id[idx], self.hhid[idx]
        else:
            return self.X[idx], self.survey_id[idx], self.hhid[idx]
