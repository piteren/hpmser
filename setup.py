from setuptools import setup, find_packages

# reads requirements
def get_requirements():
    with open('requirements.txt') as file:
        lines = [l[:-1] if l[-1]=='\n' else l for l in file.readlines()]
        return lines


setup(
    name=               'hpmser',
    version=            'v2.2.0',
    url=                'https://github.com/piteren/hpmser.git',
    author=             'Piotr Niewinski',
    author_email=       'pioniewinski@gmail.com',
    description=        'Hyper Parameters Search tool',
    packages=           find_packages(),
    install_requires=   get_requirements(),
    license=            'MIT')