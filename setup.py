"""This is a setup.py script to install scraper as a package."""

import os

from setuptools import setup, find_packages

SETUP_PTH = os.path.dirname(os.path.abspath(__file__))

readme = 'README.md'
long_description = open(readme).read()

setup(
    name='scraper',
    packages=find_packages(),
    version='0.0.1',
    description='A code which uses GPT in order to exctract unstructured information from papers.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'click',
    ],
    license='MIT',
    entry_points={
        'console_scripts': [
            'ps_search=cli.cli:paper_search',
            'ps_elsevier=cli.cli:elsevier_download',
            'ps_scrape=cli.cli:scrape',
            'ps_store=cli.cli:store'
        ],
})