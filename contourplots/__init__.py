# -*- coding: utf-8 -*-
# ContourPlots: A toolkit for generating multi-dimensional plots easily, usefully, and legibly.
# Copyright (C) 2022-2023, Sarang Bhagwat <sarang.bhagwat.git@gmail.com>
# 
# This module is under the MIT open-source license. See 
# https://github.com/sarangbhagwat/contourplots/blob/main/LICENSE
# for license details.

__version__ = '0.0.1'


# %% Initialize ContourPlots 

from . import utils

animated_contourplot = utils.animated_contourplot

__all__ = (
    'utils', 'animated_contourplot',
)
