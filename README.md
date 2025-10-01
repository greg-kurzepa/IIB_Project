# Bayesian Updating for Reduction of Soil Parameter Uncertainty Using Strain Data From Axially Loaded Piles

This repository contains the code used for my master's thesis, title above. The technical report `Report.pdf` should be read first. The presentation slides are in `Presentation.pptx`

The file that does Bayesian inference for the linear-elastic soil-pile model is `infer_linearelastic.py`.
The file that does Bayesian inference for the nonlinear API soil-pile model is `run_inference.py`.

The main idea is to use strain gauge data from existing piles to probabilistically back-analyse soil parameters, taking measurement error and other unmodelled randomness into account. Here is an example of an update to a prior soil parameter (unit weight in the top 12.5m of soil), providing a data-informed posterior:

<img width="288" height="176" alt="image" src="https://github.com/user-attachments/assets/4649f4e0-01ae-48d7-bb3b-0656cb83717a" />

This reduces uncertainty on predicted loading on the pile, which potentially allows for less material-intensive design:

<img width="672" height="746" alt="image" src="https://github.com/user-attachments/assets/f5bcb898-1299-4bab-8d5f-6cb1bc3f1e97" />
