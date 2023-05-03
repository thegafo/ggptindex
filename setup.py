from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="ggptindex",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ggptindex=ggptindex.__main__:main",
        ],
    },
    install_requires=required_packages,
)
