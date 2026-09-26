from setuptools import find_packages, setup

setup(
    name="ripe-mambaspike",
    version="1.0.0",
    description="Resolution-Independent Spiking-State-Space Interfaces for "
                "Parameter-Efficient Event-Based Vision",
    packages=find_packages(include=["ripe_mambaspike", "ripe_mambaspike.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "timm>=0.9.12",
        "einops>=0.7.0",
        "tqdm>=4.66.0",
    ],
)
