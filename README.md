## Bayesian Model Averaging using SMC^2

This repository contains the implementation code of the paper **"A sequential ensemble approach to epidemic modeling: Combining Hawkes and SEIR models using SMC^2"**.
The project introduces a novel framework that combines discrete-time Hawkes processes (DTHP) and Susceptible-Exposed-Infectious-Removed (SEIR) models for improved epidemic tracking and forecasting.

---

## Installation
To install and set up the environment for running this model, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Dhorasso/bma-smc2-dthp-seir.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---
    
## Repository Structure

- [models.py](https://github.com/Dhorasso/bma-smc2-dthp-seir/blob/main/models.py) Implements different epidemic models, including **SEIR, SEIRS, and DTHP**.

- [observation_dist.py](https://github.com/Dhorasso/bma-smc2-dthp-seir/blob/main/observation_dist.py) Contains different observation distribution functions, such as **Poisson, Normal, Normal Approximation to Negative Binomial, and Negative Binomial**.

- [smc.py](https://github.com/Dhorasso/bma-smc2-dthp-seir/blob/main/smc.py) Implements the **Bootstrap Particle Filter**, which runs in parallel for the **DTHP and SEIR models**.

- [pmmh.py](https://github.com/Dhorasso/bma-smc2-dthp-seir/blob/main/pmmh.py) Implements the **Particle Marginal Metropolis-Hastings (PMMH)** algorithm for Bayesian parameter inference.

- [smc_squared.py](https://github.com/Dhorasso/bma-smc2-dthp-seir/blob/main/smc_squared.py)  Contains the **main code for Bayesian Model Averaging (BMA)** using the **SMC^2 algorithm**.
 

---

## Example Usage

The folder [Test_bma_smc2](https://github.com/Dhorasso/bma-smc2-dthp-seir/blob/main/Test_bma_smc2) containt the code recreate the analyses conducted in the paper.
