import numpy as np
from torch.utils.data import DataLoader, Subset
from utils.datasets import HouseholdDataset

def build_loaders(
    train_csv,
    train_gt_csv,
    test_csv,
    batch_size,
    valid_survey_ids,
    max_valid_samples):
     # Full train dataset (features + targets)
    train_ds_full = HouseholdDataset(features_csv=train_csv,gt_csv=train_gt_csv)

    # test dataset
    test_ds = HouseholdDataset(features_csv=test_csv,gt_csv=None)

    survey_ids = train_ds_full.survey_id
    all_indices = np.arange(len(train_ds_full))

     # Train / valid split
    if valid_survey_ids is not None:
        # survey-aware split (CORRECT)
        survey_mask = np.isin(survey_ids, valid_survey_ids)
        survey_indices = all_indices[survey_mask]

    
        if max_valid_samples is not None:
            valid_idx = survey_indices[-max_valid_samples:]
            train_idx = np.setdiff1d(all_indices, valid_idx)
        else:
            valid_idx = survey_indices
            train_idx = np.setdiff1d(all_indices, valid_idx)

    train_ds = Subset(train_ds_full, train_idx)
    valid_ds = Subset(train_ds_full, valid_idx)

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, valid_loader, test_loader
