from setuptools import setup, find_packages

setup(
    name="TangleNAS",
    version="0.1.0",
    description="A project for neural architecture search with mixed operations.",
    author="Hoang-Loc La",
    author_email="laloc2496@gmail.com",
    url="https://github.com/hoanglocla9/TangleNAS",
    packages=find_packages(),
    install_requires=[
        "torch",
        "einops",
        "transformers",
        "datasets",
        "wandb",
        # Add other dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
