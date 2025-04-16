from setuptools import find_packages, setup

setup(
    name="ChurnPrediction",
    version="0.0.1",
    author="MD Tanveer",
    author_email="tanveersiddiqui243@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"
    ]
)

