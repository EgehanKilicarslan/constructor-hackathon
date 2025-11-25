# Reasoning Granularity Framework Demo

This project provides a simplified demonstration of the "Reasoning Granularity Framework" concept, inspired by the original research project "Unlocking the Boundaries of Thought: A Reasoning Granularity Framework to Quantify and Optimize Chain-of-Thought".

It includes a Python script (`main.py`) that simulates an evaluation process for different reasoning scenarios, using command-line arguments for flexibility.

## 🚀 Getting Started

### Prerequisites

*   Docker (recommended for easy setup)
*   Python 3.11+ (if running locally without Docker)

### 1. Build the Docker Image

Navigate to the project root directory (where `Dockerfile` is located) and build the Docker image:

```bash
docker build -t reasoning-granularity-demo .
```

### 2. Run the Evaluation

You can run the evaluation script inside the Docker container.

#### Default Evaluation (e.g., CoT split)

```bash
docker run reasoning-granularity-demo evaluate --data_split CoT
```

#### Custom Evaluation with Parameters

To simulate a custom evaluation, you can provide specific parameters. The `main.py` script will generate dummy data if the `result_path` does not exist.

```bash
docker run reasoning-granularity-demo evaluate --data_split custom --K 0.301 --K2 0.92 --mode nl --result_path /app/mock_predictions.jsonl
```

*Note: The `mock_predictions.jsonl` file is not provided in this demo. The script will generate dummy data if it's not found, demonstrating the file loading logic.*

#### Running Locally (without Docker)

If you prefer to run the script directly on your machine:

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the script:**
    ```bash
    python main.py evaluate --data_split CoT
    python main.py evaluate --data_split custom --K 0.301 --K2 0.92 --mode nl --result_path mock_predictions.jsonl
    ```

## 💡 Project Structure

```
.
├── main.py             # The main Python script for evaluation
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration for containerization
└── README.md           # Project documentation
```

## ✒️ Reference

This demo is inspired by the following research paper. If you find the concepts useful, please consider citing the original work:

```
@inproceedings{chen-etal-2024-rg,
    title = "Unlocking the Boundaries of Thought: A Reasoning Granularity Framework to Quantify and Optimize Chain-of-Thought",
    author = "Chen, Qiguang  and
      Qin, Libo  and
      Jiaqi, Wang  and
      Jinxuan, Zhou  and
      Che, Wanxiang",
    booktitle = "Proc. of NeurIPS",
    year = "2024",
}
```

## 📧 Contact

For questions or suggestions regarding the original research, please refer to the contact information in the original project's README. For this demo, you can raise an issue on the repository where this code was generated.