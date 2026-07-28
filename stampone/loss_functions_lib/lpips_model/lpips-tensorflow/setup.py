#!/usr/bin/env python
"""
Vendored packaging script for alexlee-gk/lpips-tensorflow (the lpips_tf
module used elsewhere in loss_functions_lib/lpips_model/lpips-tensorflow).
"""

from distutils.core import setup

setup(
      name='lpips-tf',
      description='Tensorflow port for the Learned Perceptual Image Patch Similarity (LPIPS) metric',
      author='Alex Lee',
      url='https://github.com/alexlee-gk/lpips-tensorflow/',
      py_modules=['lpips_tf']
)
