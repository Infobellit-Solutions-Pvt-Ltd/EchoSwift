from setuptools import setup, find_packages

with open("PIP_Package.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="echoswift",
    version="2.0.2",
    author="Infobell AI Team",
    author_email="sarthak@infobellit.com",
    description="LLM Inference Benchmarking Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Infobellit-Solutions-Pvt-Ltd/EchoSwift",
    packages=find_packages(include=['echoswift', 'echoswift.*']),
    include_package_data=True,
    package_data={
        'echoswift': ['utils/*.py', '*.py'],
    },
    install_requires=[
        "click",
        "pyyaml",
        "tqdm",
        "pandas",
        "matplotlib",
        "locust",
        "transformers",
        "datasets",
        "tabulate",
        "keyboard",
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