# Constructor Hackathon - Automated Demo Generator

This project is an intelligent agentic workflow designed to analyze research papers and GitHub repositories, automatically generating runnable demo projects. It leverages Large Language Models (LLMs) to understand complex technical concepts and synthesize functional code, Docker configurations, and documentation.

## 🚀 Features

- **Automated Analysis**: Ingests and analyzes academic articles (PDFs) and GitHub repository structures.
- **Code Generation**: Automatically generates a fully functional `main.py` demo script based on the analysis.
- **Containerization**: Creates a `Dockerfile` to ensure the generated demo is reproducible and isolated.
- **Documentation**: Generates a comprehensive `README.md` for the specific generated demo.
- **Agentic Workflow**: Built using **LangGraph** to manage state and orchestrate the analysis and generation steps.

## 🛠️ Tech Stack

- **Python 3.13+**
- **LangChain & LangGraph**: For orchestration and agent logic.
- **Google Gemini**: Used as the underlying ChatModel (via `langchain-google-genai`).
- **Pandas**: For data manipulation and processing project CSVs.
- **PyMuPDF**: For PDF text extraction from research papers.

## 📂 Project Structure

```text
.
├── assets/                 # Contains dataset CSVs (projects.csv, projects_info.csv)
├── results/                # Output directory for generated demo projects
├── src/
│   ├── agent/
│   │   └── graph.py        # Main LangGraph agent definition and logic
│   └── service/            # Service layers (ArticleAnalyser, ChatModel, etc.)
├── pyproject.toml          # Project dependencies and configuration
└── README.md               # This file
```

## ⚙️ Setup & Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd constructor-hackathon
    ```

2.  **Install dependencies:**
    This project uses pyproject.toml. You can install dependencies using pip:

    ```bash
    pip install -e .
    ```

3.  **Environment Configuration:**
    Create a .env file in the root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    CSV_FILE_PATH=path_to_csv
    ```

## 🏃 Usage

The core logic resides in graph.py. The agent performs the following steps:

1.  **Ingestion**: Reads project metadata from assets.
2.  **Analysis**: Downloads and processes linked articles or repositories.
3.  **Generation**: Uses the LLM to generate a JSON solution containing:
    - `main.py`
    - requirements.txt
    - `Dockerfile`
    - README.md
4.  **Output**: Writes the generated files to the results directory.

To run the agent (ensure you are in the root directory):

```bash
python src/__main__.py --link {link}
```

_(Note: Adjust the command if you have a specific entry point script like `main.py`)_
