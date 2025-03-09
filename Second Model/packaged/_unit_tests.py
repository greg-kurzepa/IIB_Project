import numpy as np
import pandas as pd

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

def run_tests():
    test_solver()

def test_solver():
    # round numpy array to p sigfigs. https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy
    def signif(x, p):
        x = np.asarray(x)
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
        mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

    # Test 1: run _model_springs.solve_springs with a preset pile and soil to check the output matches my benchmark
    benchmark_pile = _pile_and_soil.Pile(R=0.15, L=30, W=2.275e3, E=20e9)
    benchmark_soil_layer = _pile_and_soil.SoilLayer(alpha=0.4, gamma=20e3, N_c=9, s_u0=30e3, rho=4.8e3, base_depth=benchmark_pile.L)
    benchmark_soil = _pile_and_soil.Soil([benchmark_soil_layer])
    benchmark_P = 200e3 # top axial load
    benchmark_N = 100 # number of nodes along pile
    benchmark_z, benchmark_F, benchmark_strain, benchmark_u = _model_springs.solve_springs(benchmark_pile, benchmark_soil, benchmark_P, benchmark_N)

    benchmark_df = pd.read_csv("Second Model\\packaged\\benchmark_data.csv", skiprows=4, float_precision="round_trip")
    assert all([
        np.isclose(signif(benchmark_F, 3), benchmark_df["F"].to_numpy()).all(),
        np.isclose(signif(benchmark_strain, 3), benchmark_df["strain"].to_numpy()).all(),
        np.isclose(signif(benchmark_u, 3), benchmark_df["u"].to_numpy()).all()])