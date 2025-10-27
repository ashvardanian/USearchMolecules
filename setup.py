from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="usearch-molecules",
    version="1.0.0",
    description="Large Chem-Informatics dataset of 7B+ molecules with binary fingerprints for drug discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ash Vardanian",
    author_email="ash.vardanian@unum.cloud",
    url="https://github.com/ashvardanian/usearch-molecules",
    license="Apache-2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.9",
    keywords="molecules, chemistry, drug-discovery, fingerprints, similarity-search, usearch",
)
