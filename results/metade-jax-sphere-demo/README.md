# MetaDE JAX Sphere Optimization Demo

This project provides a Dockerized demonstration of the MetaDE framework using its JAX backend to optimize hyperparameters for a Differential Evolution (DE) algorithm. The DE algorithm is then used to solve the classic Sphere benchmark function.

## Project Description

MetaDE is an advanced evolutionary framework that dynamically optimizes the strategies and hyperparameters of Differential Evolution (DE) through meta-level evolution. This demo focuses on a specific example from the MetaDE project, demonstrating how to set up and run a hyperparameter optimization task for a simple numerical problem (Sphere function) using JAX.

## Features

*   **Meta-level Evolution**: Uses DE at a meta-level to evolve hyperparameters and strategies of DE applied at a problem-solving level.
*   **Parameterized DE (PDE)**: A customizable variant of DE that offers dynamic mutation and crossover strategies.
*   **JAX Backend**: Leverages JAX for high-performance numerical computation.
*   **Containerized**: Provided with a `Dockerfile` for easy setup and execution in an isolated environment.

## Demo Overview (`main.py`)

The `main.py` script sets up a MetaDE workflow to optimize the parameters of a base Differential Evolution (DE) algorithm. This base DE algorithm is then tasked with finding the minimum of the 10-dimensional Sphere function.

1.  **Problem Definition**: A 10-dimensional Sphere function is used as the base optimization problem.
2.  **Meta-Algorithm (Outer Loop)**: A DE algorithm (`evolver`) is configured to search for optimal hyperparameters for the base DE.
3.  **Base Algorithm (Inner Loop)**: A `ParamDE` (Parameterized DE) algorithm is used to solve the Sphere problem. Its hyperparameters (e.g., differential weight, cross probability, mutation strategies) are the target of the outer DE.
4.  **Workflow**: The `StdWorkflow` orchestrates the meta-evolutionary process, where the outer DE iteratively proposes hyperparameters, and the inner `ParamDE` evaluates their performance on the Sphere function.
5.  **Output**: The script prints the best fitness found by the meta-optimization process, indicating how well the evolved hyperparameters performed.

## Getting Started with Docker

To run this demo, you need Docker installed on your system. Follow these steps:

1.  **Clone the repository** (or create the files manually):
    ```bash
    # Assuming you have the project files in a directory named metade-jax-sphere-demo
    cd metade-jax-sphere-demo
    ```

2.  **Build the Docker image**:
    ```bash
    docker build -t metade-jax-sphere-demo .
    ```

3.  **Run the Docker container**:
    ```bash
    docker run metade-jax-sphere-demo
    ```

    You will see progress updates from `tqdm` and the final best fitness printed to your console.

## Original MetaDE Project Information

This demo is based on the [MetaDE project](https://github.com/EMI-Group/metade), which is compatible with the [EvoX framework](https://github.com/EMI-Group/evox).

### RL Tasks Visualization (from original README)

Using the MetaDE algorithm to solve RL tasks.

The following animations show the behaviors in Brax environments:

<table width="81%">
  <tr>
    <td width="27%">
      <img width="200" height="200" style="display:block; margin:auto;"  src="https://raw.githubusercontent.com/EMI-Group/metade/main/assets/hopper.gif">
    </td>
    <td width="27%">
      <img width="200" height="200" style="display:block; margin:auto;" src="https://raw.githubusercontent.com/EMI-Group/metade/main/assets/swimmer.gif">
    </td>
    <td width="27%">
      <img width="200" height="200" style="display:block; margin:auto;" src="https://raw.githubusercontent.com/EMI-Group/metade/main/assets/reacher.gif">
    </td>
  </tr>
  <tr>
    <td align="center">
      Hopper
    </td>
    <td align="center">
      Swimmer
    </td>
    <td align="center">
      Reacher
    </td>
  </tr>
</table>

-   **Hopper**: Aiming for maximum speed and jumping height.  
-   **Swimmer**: Enhancing movement efficiency in fluid environments.  
-   **Reacher**: Moving the fingertip to a random target.

### Citing MetaDE

If you use MetaDE in your research, please cite the original paper:

```
@article{metade,
  title = {{MetaDE}: Evolving Differential Evolution by Differential Evolution},
  author = {Chen, Minyang and Feng, Chenchen and Cheng, Ran},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = 2025,
  doi = {10.1109/TEVC.2025.3541587}
}
```
