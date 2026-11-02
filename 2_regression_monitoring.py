import numpy as np
import pandas as pd
from sklearn import datasets

from evidently import Report
from evidently.core.datasets import Dataset, DataDefinition, Regression
from evidently.presets import RegressionPreset
from evidently.ui.workspace import Workspace


def get_data():
    """
    Loads the diabetes dataset from sklearn and creates a DataFrame.
    """
    diabetes_data = datasets.load_diabetes()
    diabetes_df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
    diabetes_df["target"] = diabetes_data.target
    diabetes_df["prediction"] = diabetes_df["target"].values + np.random.normal(
        0, 3, diabetes_df.shape[0]
    )
    return diabetes_df


def prepare_data(dataframe, ref_sample_size, curr_sample_size):
    """
    Prepares reference and current datasets for comparison.
    """
    reference_data = dataframe.sample(n=ref_sample_size, replace=False)
    current_data = dataframe.sample(n=curr_sample_size, replace=False)
    return reference_data, current_data


def generate_regression_performance_report(reference_data, current_data):
    """
    Generates a regression performance report using Evidently.
    """
    # Define regression task mapping (Modern way)
    # Regression metrics need to know which columns are target and prediction
    definition = DataDefinition(
        regression=[Regression(target="target", prediction="prediction")]
    )

    # Wrap pandas DataFrames into Dataset objects
    reference_dataset = Dataset.from_pandas(reference_data, data_definition=definition)
    current_dataset = Dataset.from_pandas(current_data, data_definition=definition)

    # Create a report instance for regression with RegressionPreset()
    regression_performance_report = Report(metrics=[RegressionPreset()])

    # Run the report and return the snapshot
    snapshot = regression_performance_report.run(
        reference_data=reference_dataset, current_data=current_dataset
    )
    return snapshot


def add_report_to_workspace(workspace, project_name, project_description, snapshot):
    """
    Adds a snapshot to an existing or new project in a workspace.
    """
    # Check if project already exists
    projects = workspace.search_project(project_name)

    # Create a new project if it doesn't exist
    if not projects:
        project = workspace.create_project(project_name)
        project.description = project_description
    else:
        project = projects[0]

    # Add snapshot to the project
    workspace.add_run(project.id, snapshot)
    print(f"New snapshot added to project {project_name}")


if __name__ == "__main__":
    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "regression_monitoring"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Obtener el conjunto de datos de diabetes
    diabetes_df = get_data()

    # Prepare reference and current datasets by taking a sample of 50 rows
    reference_data, current_data = prepare_data(
        diabetes_df, ref_sample_size=50, curr_sample_size=50
    )

    # Generate the regression performance report
    regression_snapshot = generate_regression_performance_report(
        reference_data, current_data
    )

    # Create and Add report to workspace
    workspace = Workspace.create(WORKSPACE_NAME)
    add_report_to_workspace(
        workspace, PROJECT_NAME, PROJECT_DESCRIPTION, regression_snapshot
    )
