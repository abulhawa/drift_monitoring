import numpy as np
from sklearn import datasets

from evidently import Report
from evidently.core.datasets import Dataset
from evidently.core.datasets import DataDefinition
from evidently.presets import DataDriftPreset

adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
# Transform into dataframe
adult = adult_data.frame

selected = ["Some-college", "HS-grad", "Bachelors"]

adult_ref = adult[~adult.education.isin(selected)].copy()
adult_cur = adult[adult.education.isin(selected)].copy()

adult_cur.iloc[:2000, 3:5] = np.nan

definition = DataDefinition(
    numerical_columns=[
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ],
    categorical_columns=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "class",
    ],
)

reference_dataset = Dataset.from_pandas(adult_ref, data_definition=definition)
current_dataset = Dataset.from_pandas(adult_cur, data_definition=definition)

report = Report(metrics=[DataDriftPreset()])

snapshot = report.run(current_data=current_dataset, reference_data=reference_dataset)

snapshot.save_json("data_drift_report.json")
snapshot.save_html("data_drift_report.html")