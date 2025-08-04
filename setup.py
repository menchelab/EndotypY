from setuptools import setup, find_packages

setup(
    name='EndotypY',
    version='0.1',
    packages=find_packages(),
    install_requires=[
       'numpy',
        'pandas',
        'gprofiler-official',
        'networkx',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'scipy',
        'tqdm',
        'mygene',
        'gseapy',
        'threadpoolctl',
        'biopython',

    ],
    entry_points={
        'console_scripts': [
            # Add your command line scripts here
        ],
    },
    author='Your Name',
    author_email='mathilde.meyenberg@univie.ac.at',
    description='A short description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/EndotypY',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
