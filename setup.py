"""Setup script for clinical risk modeling package."""
from setuptools import setup, find_packages

with open("README_UPDATED.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="clinical-risk-modeling",
    version="1.0.0",
    author="João Bentes",
    author_email="",
    description="Clinical Risk Modeling with Interpretability and Bias Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JB5000/04_clinical_risk_modeling_interpretability_bias",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "clinical-train=train:main",
            "clinical-evaluate=evaluate:main",
        ],
    },
)
