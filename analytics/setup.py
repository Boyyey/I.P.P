# setup.py
from setuptools import setup, find_packages

setup(
    name="intelligent_performance_predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.24.0',
        'pandas>=2.0.0',
        'plotly>=5.0.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
        'psutil>=5.9.0',
        'pynput>=1.7.0',
        'nltk>=3.8.0',
        'textblob>=0.17.0',
    ],
    python_requires='>=3.8',
)