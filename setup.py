from setuptools import setup, find_packages

setup(
    name='echoswift',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'locust',
        'pandas',
        'transformers',
        'datasets',
        'tqdm',
        'PyYAML',
    ],
    entry_points={
        'console_scripts': [
            'echoswift=echoswift.cli:cli',
        ],
    },
)