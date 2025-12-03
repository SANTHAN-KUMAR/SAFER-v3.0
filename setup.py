"""
SAFER v3.0 Package Setup.

Scalable Aerospace Failure Estimation Runtime.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''

# Read requirements
req_path = Path(__file__).parent / 'requirements.txt'
if req_path.exists():
    with open(req_path, 'r') as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.2.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
    ]

setup(
    name='safer_v3',
    version='3.0.0',
    author='SAFER Development Team',
    description='Scalable Aerospace Failure Estimation Runtime - Turbofan Engine Prognostics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/safer-v3',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'mypy>=1.0.0',
            'flake8>=6.0.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
        'viz': [
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'safer-train-mamba=safer_v3.scripts.train_mamba:main',
            'safer-train-baselines=safer_v3.scripts.train_baselines:main',
        ],
    },
)
