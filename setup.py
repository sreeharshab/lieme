from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lieme",
    version="0.2.0",
    author="Sree Harsha Bhimineni",
    author_email="bsreeharsha@g.ucla.edu",
    description="Li-ion Intercalation Electrode Materials Exploration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sreeharshab/lieme",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.11.0",
    install_requires=[
        "tpot>=0.12.2",
        "ase>=3.23.0",
        "pymatgen>=2025.1.9",
        "matminer>=0.9.3",
        "mp-api>=0.45.3",
        "openpyxl",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
)