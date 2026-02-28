"""
CryptoLab Engine — Setup & Cython Build
"""
from setuptools import setup, find_packages

# Cython extensions (optional — falls back to pure Python)
try:
    from Cython.Build import cythonize
    import numpy as np
    ext_modules = cythonize([
        "indicators/ehlers_cy.pyx",
        "indicators/gaussian_cy.pyx",
        "core/engine_cy.pyx",
    ], compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'language_level': "3",
    })
    include_dirs = [np.get_include()]
except ImportError:
    ext_modules = []
    include_dirs = []
    print("⚠ Cython not found — building pure Python version")

setup(
    name="cryptolab",
    version="0.1.0",
    description="Backtesting & ML Engine for Crypto/Stock Perpetual Futures",
    packages=find_packages(),
    ext_modules=ext_modules,
    include_dirs=include_dirs,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "aiohttp>=3.9",
        "pyarrow>=14.0",
        "pyyaml>=6.0",
        "click>=8.1",
        "rich>=13.0",
        "tabulate>=0.9",
    ],
    extras_require={
        "optimize": ["optuna>=3.5", "deap>=1.4"],
        "ml": ["scikit-learn>=1.3", "lightgbm>=4.0"],
        "dashboard": ["streamlit>=1.30", "plotly>=5.18"],
        "cython": ["cython>=3.0"],
        "dev": ["pytest>=7.4", "hypothesis>=6.0"],
    },
    entry_points={
        "console_scripts": [
            "cryptolab=cli:main",
        ],
    },
)
