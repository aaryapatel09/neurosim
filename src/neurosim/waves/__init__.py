"""Wave equation / acoustics module.

Provides 2D acoustic wave equation solvers with heterogeneous media
support, multiple source types, and absorbing boundary conditions.
"""

from neurosim.waves.acoustic import AcousticSolver, GaussianPulse, PointSource

__all__ = [
    "AcousticSolver",
    "GaussianPulse",
    "PointSource",
]
