from setuptools import setup, find_packages

from pypaq.lipytools.files import get_requirements


setup(
    name=               'hpmser',
    version=            'v2.2.1',
    url=                'https://github.com/piteren/hpmser.git',
    author=             'Piotr Niewinski',
    author_email=       'pioniewinski@gmail.com',
    description=        'Hyper Parameters Search tool',
    packages=           find_packages(),
    install_requires=   get_requirements(),
    license=            'MIT')