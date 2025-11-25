# Generals.io RL Demo

This project provides a basic demonstration of the `generals-bots` library, a fast-paced strategy environment for reinforcement learning based on the popular web game Generals.io.

## Project Description

The `generals-bots` library offers a customizable and efficient platform for developing AI agents for Generals.io. This demo showcases how to set up a simple simulation using the `PettingZooGenerals` environment, where two pre-built agents – a `RandomAgent` and an `ExpanderAgent` – compete against each other. The simulation runs headlessly, printing game progress to the console.

## Features

*   **`main.py`**: Initializes a `PettingZooGenerals` environment with a fixed grid, runs a game loop for a set number of steps, and prints the state of the agents (land, army, rewards) at intervals.
*   **`generals-bots`**: Leverages the `generals-bots` library for environment simulation and agent interaction.
*   **Containerized**: Includes a `Dockerfile` for easy setup and execution in a Docker container.

## Project Structure

```
. 
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## How to Run

### 1. Locally (Python)

1.  **Prerequisites**:
    *   Python 3.11+ installed.
    *   `pip` for package management.

2.  **Clone the repository (or create files manually)**:
    ```bash
    # If you have the files in a directory named generals_io_rl_demo
    cd generals_io_rl_demo
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you wish to enable GUI rendering (by changing `render_mode=None` to `render_mode="human"` in `main.py`), you might need additional system libraries for Pygame (e.g., `libsdl2-2.0-0`, `libsdl2-image-2.0-0`, etc. on Debian/Ubuntu-based systems).* 

4.  **Run the application**:
    ```bash
    python main.py
    ```
    You will see console output detailing the game's progress.

### 2. With Docker

1.  **Prerequisites**:
    *   Docker installed and running on your system.

2.  **Clone the repository (or create files manually)**:
    ```bash
    # If you have the files in a directory named generals_io_rl_demo
    cd generals_io_rl_demo
    ```

3.  **Build the Docker image**:
    ```bash
    docker build -t generals-io-rl-demo .
    ```

4.  **Run the Docker container**:
    ```bash
    docker run generals-io-rl-demo
    ```
    The simulation will run inside the container, and its output will be displayed in your terminal.