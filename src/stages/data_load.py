import pandas as pd
from sklearn.datasets import load_iris
from typing import Text
import yaml

def data_load(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data = load_iris(as_frame=True)
    dataset = data.frame

    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]
    # Save
    dataset.to_csv(config['data']['dataset_csv'], index=False)