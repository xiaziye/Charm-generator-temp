"""Continuous configurable data generation for paper 1, conclusion 1."""
from .generator import ContinuousDesign, build_design, generate, simulate
from .schema import GenerationConfig, load_config

__all__ = ["ContinuousDesign", "GenerationConfig", "build_design", "generate", "load_config", "simulate"]
