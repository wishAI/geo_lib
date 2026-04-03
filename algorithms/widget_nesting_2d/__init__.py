"""2D widget nesting with hole-aware polygon placement."""

from .problem import load_problem, save_solution
from .solver import SolverConfig, solve_problem

__all__ = ["SolverConfig", "load_problem", "save_solution", "solve_problem"]
