"""
Setup configuration for AlphaMind backend.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="alphamind-backend",
    version="1.0.0",
    author="AlphaMind Team",
    author_email="contact@alphamind.ai",
    description="Institutional-Grade Quantitative AI Trading System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantsingularity/AlphaMind",
    packages=find_packages(exclude=["tests", "tests.*", "research", "research.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "alphamind-api=api.main:main",
        ],
    },
)
