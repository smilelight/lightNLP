from distutils.core import setup
import setuptools

with open('./README.md', 'r', encoding='utf8') as f:
    long_description = f.read()

with open('./requirements.txt', 'r', encoding='utf8') as f:
    install_requires = list(map(lambda x: x.strip(), f.readlines()))

setup(
    name='lightNLP',
    version='0.2.4.1',
    description="lightsmile's nlp library",
    author='lightsmile',
    author_email='iamlightsmile@gmail.com',
    url='https://github.com/smilelight/lightNLP',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries'
    ],
)