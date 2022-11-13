# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 20:30:00 2022

@author: sarangbhagwat
"""

from setuptools import setup

setup(
    name='contourplots',
    version='0.0.1',    
    description='A toolkit for generating multi-dimensional plots easily, usefully, and legibly.',
    url='https://github.com/sarangbhagwat/contourplots',
    author='Sarang S. Bhagwat',
    author_email='sarang.bhagwat.git@gmail.com',
    license='MIT',
    # packages=['contourplots'],
    install_requires=['matplotlib>=3.5.2',
                      'numpy>=1.23.4',       
                      'imageio>=2.19.3',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9'
    ],
)
