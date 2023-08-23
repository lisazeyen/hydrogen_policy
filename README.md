# Hydrogen policy

Hydrogen policy is an open model to model hydrogen production with different
regulation standards in a selected country in Europe.


1. **Setup and Environment**:

   I generally recommend using `mamba` over `conda`. You can find the installation instructions for `mamba` [here](https://mamba.readthedocs.io/en/latest/installation.html).

   From your terminal, run the following commands:

   ```bash
   git clone https://github.com/lisazeyen/hydrogen_policy.git
   mamba env create -f envs/environment.yaml
   conda activate hourly
   ```

2. **Running the Workflow**:

   To execute the workflow, use the following command:

   ```bash
   snakemake plot_all -j8
   ```

3. **Configuration**:

   You can modify assumptions in the `config` (e.g., country, storage types, demand volume, CO2 limits, etc.).

---
