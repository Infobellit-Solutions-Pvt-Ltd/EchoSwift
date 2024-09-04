from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="echoswift",
    version="0.1.1",
    author="Infobell AI Team",
    author_email="akhil@infobellit.com",
    description="LLM Inference Benchmarking Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Infobellit-Solutions-Pvt-Ltd/EchoSwift",
    packages=find_packages(include=['echoswift', 'echoswift.*']),
    include_package_data=True,
    install_requires=[
        "click",
        "pyyaml",
        "tqdm",
        "pandas",
        "matplotlib",
        "locust",
        "transformers",
        "datasets",
    ],
    entry_points={
        "console_scripts": [
            "echoswift=echoswift.cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)