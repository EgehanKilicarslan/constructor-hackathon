# LLaVA-MORE Context Sampler Demo

This project provides a simplified Python demonstration of the context sampling logic found within the `LLaVA-MORE` project, specifically inspired by the `samplers.py` file. It showcases how few-shot examples are selected and formatted to create a context for a language model, using mock data and a simplified task definition.

## Project Structure

*   `main.py`: The core Python script demonstrating the sampling logic.
*   `requirements.txt`: Lists the necessary Python dependencies.
*   `Dockerfile`: Enables containerization of the application.
*   `README.md`: This file, providing an overview and instructions.

## `main.py` Overview

The `main.py` script includes the `ContextSampler` and `FirstNSampler` classes, adapted directly from the `LLaVA-MORE` codebase (`src/lmms_eval/api/samplers.py`). To make the demo runnable without the full `lmms_eval` framework, it includes mock implementations for `MockTaskConfig` and `MockTask`.

The script performs the following steps:
1.  **Mocks Data**: Creates a simple `datasets.Dataset` object with mock questions and answers.
2.  **Mocks Task**: Instantiates a `MockTask` object, which provides the necessary configuration and methods (`doc_to_text`, `doc_to_target`) that the samplers expect.
3.  **Demonstrates `ContextSampler`**: Shows how `ContextSampler` randomly selects few-shot examples and formats them into a coherent context string.
4.  **Demonstrates `FirstNSampler`**: Illustrates how `FirstNSampler` selects the first `n` examples in order, which is useful for tasks with canonical few-shot examples.

The script accepts command-line arguments for `num_fewshot` (number of examples) and `seed` (for reproducibility).

## Getting Started

### Prerequisites

*   Python 3.8+ (or use Docker)
*   `pip` (Python package installer)

### Local Installation and Run

1.  **Clone the repository (or create the files manually):**
    ```bash
    # If this were a real repo, you'd clone it.
    # For this demo, assume you have main.py and requirements.txt
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the demo script:**
    ```bash
    python main.py
    # Or with custom parameters:
    python main.py --num-fewshot 3 --seed 123
    ```

### Running with Docker

Docker provides a consistent environment to run the application.

1.  **Ensure Docker is installed:**
    If you don't have Docker installed, follow the instructions on the [official Docker website](https://docs.docker.com/get-docker/).

2.  **Build the Docker image:**
    Navigate to the directory containing `Dockerfile`, `main.py`, and `requirements.txt`, then run:
    ```bash
    docker build -t llava-more-sampler-demo .
    ```

3.  **Run the Docker container:**
    ```bash
    docker run llava-more-sampler-demo
    # Or with custom parameters:
    docker run llava-more-sampler-demo python main.py --num-fewshot 3 --seed 123
    ```

## Original Project Context

This demo is inspired by the `LLaVA-MORE` project, a research effort by AImageLab focusing on comparative studies of LLMs and visual backbones for enhanced visual instruction tuning. The `samplers.py` file, from which the core logic for this demo is derived, is part of their `lmms_eval` evaluation framework, responsible for preparing context for multimodal language models.

For more details on the original `LLaVA-MORE` project, please refer to their [GitHub repository](https://github.com/aimagelab/LLaVA-MORE) and [paper](https://arxiv.org/abs/2503.15621).
