# -*- coding: utf-8 -*-
# ContourPlots: A toolkit for generating multi-dimensional plots easily, usefully, and legibly.
# Copyright (C) 2022-2023, Sarang Bhagwat <sarang.bhagwat.git@gmail.com>
# 
# This module is under the MIT open-source license. See 
# https://github.com/sarangbhagwat/contourplots/blob/main/LICENSE
# for license details.

__version__ = '0.3.4'
__author__ = 'Sarang S. Bhagwat'

# %% Initialize ContourPlots 

from . import utils

animated_contourplot = utils.animated_contourplot
animated_barplot = utils.animated_barplot
box_and_whiskers_plot = utils.box_and_whiskers_plot
stacked_bar_plot = utils.stacked_bar_plot
animated_stacked_barplot = utils.animated_stacked_barplot

__all__ = (
    'utils', 
    'animated_contourplot', 
    'animated_barplot',
    'animated_stacked_barplot',
    'box_and_whiskers_plot', 
    'stacked_bar_plot',
)
