from setuptools import setup, find_packages

setup(
    name="polydiff",
    version="0.1.0",
    packages=find_packages(include=["PolyDiff", "PolyDiff.*"]),
    install_requires=[
        "torch",
        "omegaconf",
        "hydra-core",
    ],
)
