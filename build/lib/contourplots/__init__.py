# -*- coding: utf-8 -*-
# ContourPlots: A toolkit for generating multi-dimensional plots easily, usefully, and legibly.
# Copyright (C) 2022-2023, Sarang Bhagwat <sarang.bhagwat.git@gmail.com>
# 
# This module is under the MIT open-source license. See 
# https://github.com/sarangbhagwat/contourplots/blob/main/LICENSE
# for license details.

__version__ = '0.2.1'
__author__ = 'Sarang S. Bhagwat'

# %% Initialize ContourPlots 

from . import utils

animated_contourplot = utils.animated_contourplot
box_and_whiskers_plot = utils.box_and_whiskers_plot
stacked_bar_plot = utils.stacked_bar_plot

__all__ = (
    'utils', 
    'animated_contourplot', 
    'box_and_whiskers_plot', 
    'stacked_bar_plot',
)
