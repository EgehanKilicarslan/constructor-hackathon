import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel


class ProjectInfo(BaseModel):
    title: str
    github_url: str
    main_categort: str
    secondary_category: str
    tags: list[str]
    paper_url: str
    specific_url: str
    documentation_url: str
    dependencies_count: int
    special_features: str
    entry_points: str
    execution_command: str
    extrenal_credentials: str | None = None
    dataset_dependencies: str | None = None
    terminate: bool = False


class CSVAnalyser:
    def __init__(self, file_path: Path | None = None) -> None:
        load_dotenv()
        self.df = pd.read_csv(file_path or os.getenv("CSV_FILE_PATH", "assets/projects.csv"))

    def search_project(self, query: str) -> ProjectInfo | None:
        query = query.strip()

        mask = (
            self.df["Paper URL"].str.contains(query, case=False, na=False)
            | self.df["Specific URL"].str.contains(query, case=False, na=False)
            | self.df["Github URL"].str.contains(query, case=False, na=False)
        )

        results = self.df[mask]

        if not results.empty:
            return ProjectInfo(
                title=results.iloc[0]["Title"],
                github_url=results.iloc[0]["Github URL"],
                main_categort=results.iloc[0]["Main Category"],
                secondary_category=results.iloc[0]["Secondary Category"],
                tags=results.iloc[0]["Tags"],
                paper_url=results.iloc[0]["Paper URL"],
                specific_url=results.iloc[0]["Specific URL"],
                documentation_url=results.iloc[0]["Documentation"],
                dependencies_count=results.iloc[0]["Dependencies"],
                special_features=results.iloc[0]["Special features"],
                entry_points=results.iloc[0]["Entry Points (Scripts)"],
                execution_command=results.iloc[0]["Execution Command"],
                extrenal_credentials=results.iloc[0].get("External Credentials"),
                dataset_dependencies=results.iloc[0].get("Dataset Dependencies"),
            )
        return None
