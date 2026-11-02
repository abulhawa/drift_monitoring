import pandas as pd
import numpy as np
import zipfile

from evidently import Report
from evidently.core.datasets import Dataset, DataDefinition
from evidently.presets import DataSummaryPreset
from evidently.metrics import MeanValue
from evidently.tests import eq, Reference
from evidently.ui.workspace import Workspace


def load_data_from_zip(zip_path, sample_size):
    """
    Load a sample of data from a ZIP file and return as a DataFrame.
    """
    np.random.seed(42)

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        # Get the first file in the ZIP archive
        file_name = zip_file.namelist()[0]
        # Read a sample of the file as a DataFrame
        df = pd.read_csv(zip_file.open(file_name), index_col=0).sample(
            n=sample_size, random_state=42
        )
    return df


def split_data_by_month(df, month=6):
    """
    Split data into reference and current datasets based on month.
    """
    # Filter data based on the month
    month_range = df["Month"] >= month
    ref_data = df[~month_range]
    curr_data = df[month_range]
    return ref_data, curr_data


def create_report_and_run_tests(ref_data, curr_data, save_file=True):
    """
    Create and run a report for dataset quality and custom tests.
    """
    # Define data mapping
    definition = DataDefinition()

    # Wrap pandas DataFrames into Dataset objects
    reference_dataset = Dataset.from_pandas(ref_data, data_definition=definition)
    current_dataset = Dataset.from_pandas(curr_data, data_definition=definition)

    # Initialize Report with preset and custom metric test
    report = Report(
        metrics=[
            DataSummaryPreset(include_tests=True),
            MeanValue(column="ArrDelay", tests=[eq(Reference())]),
        ],
        include_tests=True,
    )

    # Ejecutar el informe y devolver la instantánea
    snapshot = report.run(
        current_data=current_dataset, reference_data=reference_dataset
    )

    # Save the report as HTML
    if save_file:
        snapshot.save_html("data_drift_suite.html")
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

    # Agregar instantánea al proyecto (add_run reemplaza a add_snapshot)
    workspace.add_run(project.id, snapshot)
    print(f"Nueva instantánea agregada al proyecto {project_name}")


if __name__ == "__main__":
    # Define constants for workspace and project details
    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "test_suites"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Path to the ZIP file
    zip_path = "./data/Delay_data.zip"

    # Define sample size
    sample_size = 10000

    # Load data sample from ZIP file
    df = load_data_from_zip(zip_path, sample_size)

    # Split data into reference and current datasets
    ref_data, curr_data = split_data_by_month(df)

    # Create and run report for dataset summary
    snapshot = create_report_and_run_tests(ref_data, curr_data)

    # Add report to workspace
    workspace = Workspace.create(WORKSPACE_NAME)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, snapshot)
