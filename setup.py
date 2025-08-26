from setuptools import setup, find_packages

setup(
    name='biotagging',
    version='0.1.0',
    author='Saroop & Aiman',
    author_email='aimankoli90@gmail.com',
    description='A package for tagging data for NER (BIO format)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Aimankoli/biotagging',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        "pandas",
        "numpy",
    ],
)