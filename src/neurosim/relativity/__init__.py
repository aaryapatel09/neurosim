"""General relativity module.

Provides geodesic integration in Schwarzschild and Kerr spacetimes,
gravitational lensing ray tracing, and special relativistic Lorentz
transformations.
"""

from neurosim.relativity.geodesic import GeodesicSolver
from neurosim.relativity.lensing import GravitationalLens
from neurosim.relativity.lorentz import (
    lorentz_boost,
    lorentz_factor,
    proper_time,
    velocity_addition,
)

__all__ = [
    "GeodesicSolver",
    "GravitationalLens",
    "lorentz_boost",
    "lorentz_factor",
    "proper_time",
    "velocity_addition",
]
