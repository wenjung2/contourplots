# -*- coding: utf-8 -*-
# ContourPlots: A toolkit for generating multi-dimensional plots easily, usefully, and legibly.
# Copyright (C) 2022-2023, Sarang Bhagwat <sarang.bhagwat.git@gmail.com>
# 
# This module is under the MIT open-source license. See 
# https://github.com/sarangbhagwat/contourplots/blob/main/LICENSE
# for license details.
"""
Created on Fri Nov 11 10:15:45 2022

@author: sarangbhagwat
"""

#%% Plot MPSP

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
import os
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, LinearLocator, FixedLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.container import BarContainer
import textwrap
import itertools
from math import ceil, floor
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom

defaults_dict ={'colors':
                {'Guest_Group_TEA_Breakdown': ['#7BBD84', '#F7C652', '#63C6CE', '#94948C', '#734A8C', '#D1C0E1', '#648496', '#B97A57', '#D1C0E1', '#F8858A', '#F8858A', ]}}

map_superscript_numbers = {
    0: '\u2070',
    1: '\u00B9',
    2: '\u00B2',
    3: '\u00B3',
    4: '\u2074',
    5: '\u2075',
    6: '\u2076',
    7: '\u2077',
    8: '\u2078',
    9: '\u2079',
    }

map_superscript_str_numbers = {str(k):v for k,v in map_superscript_numbers.items()}

def wrap_labels(ax, width, break_long_words=False, fontsize=14):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0, fontsize=fontsize, weight='bold')

def limit_contour(ax, x,y,z,clevs, xlim=None, ylim=None, **kwargs): # from https://stackoverflow.com/questions/45844745/moving-contour-labels-after-limiting-plot-size
    x,y = np.meshgrid(x,y)    
    mask = np.ones(x.shape).astype(bool)
    if xlim:
        mask = mask & (x>=xlim[0]) & (x<=xlim[1])
    if ylim:
        mask = mask & (y>=ylim[0]) & (y<=ylim[1])
    xm = np.ma.masked_where(~mask , x)
    ym = np.ma.masked_where(~mask , y)
    # breakpoint()
    zm = np.ma.masked_where(~mask , z)

    cs = ax.contour(xm,ym,zm, clevs,**kwargs)
    if xlim: ax.set_xlim(xlim) #Limit the x-axis
    if ylim: ax.set_ylim(ylim)
    ax.clabel(cs,inline=True,fmt='%3.0d')
    
def animated_contourplot(w_data_vs_x_y_at_multiple_z, # shape = z * x * y
                                  x_data,
                                  y_data,
                                  z_data,
                                  x_label, # title of the x axis
                                  y_label, # title of the y axis
                                  z_label, # title of the z axis
                                  w_label, # title of the color axis
                                  x_ticks,
                                  y_ticks,
                                  z_ticks,
                                  w_ticks, # labeled, lined contours (a subset of w_levels)
                                  w_levels, # unlabeled, filled contour areas (labeled and ticked only on color bar)
                                  x_units,
                                  y_units,
                                  z_units,
                                  w_units,
                                  w_tick_width=0.5, # width for labeled, lined contours
                                  fmt_clabel = lambda cvalue: "{:.2f}".format(cvalue), # format of contour labels
                                  gridspec_kw={'height_ratios': [1, 20]},
                                  fontname={'fontname':'Arial Unicode'},
                                  figwidth=3.9,
                                  dpi=600,
                                  cmap='viridis',
                                  extend_cmap='neither',
                                  cmap_over_color=None,
                                  cmap_under_color=None,
                                  label_over_color=None,
                                  label_under_color=None,
                                  cbar_ticks=None,
                                  z_marker_color='b',
                                  z_marker_type='v',
                                  axis_title_fonts={'size': {'x': 12, 'y':12, 'z':12, 'w':12},},
                                  gap_between_figures=20., 
                                  clabel_fontsize = 12,
                                  fps=3, # animation frames (z values traversed) per second
                                  n_loops='inf', # the number of times the animated contourplot should loop animation over z; infinite by default
                                  animated_contourplot_filename='animated_contourplot',
                                  keep_frames=False, # leaves frame PNG files undeleted after running; False by default
                                  keep_gifs=True, # saves GIF files; True by default
                                  n_minor_ticks = 1,
                                  cbar_n_minor_ticks = 4,
                                  comparison_range=[],
                                  comparison_range_hatch_pattern='///',
                                  
                                  comparison_lines = [],
                                  comparison_lines_colors='white',
                                  
                                  default_fontsize=12.,
                                  units_on_newline = (True, True, False, False), # x,y,z,w
                                  units_opening_brackets = [" [",] * 4,
                                  units_closing_brackets = ["]",] * 4,
                                  manual_clabels_regular={}, # clabel: (x,y)
                                  manual_clabels_comparison_range={},# clabel: (x,y)
                                  contourplot_facecolor=np.array([None, None, None]),
                                  text_boxes = {}, # str: (x,y)
                                  additional_points = {}, # (x,y): (markershape, markercolor, markersize)
                                  additional_vlines = [],
                                  additional_vline_colors='black',
                                  additional_vline_linestyles='dashed',
                                  additional_vline_linewidths=0.8,
                                  additional_hlines = [],
                                  additional_hline_colors='black',
                                  additional_hline_linestyles='dashed',
                                  additional_hline_linewidths=0.8,
                                  axis_tick_fontsize=12.,
                                  gaussian_filter_smoothing=False,
                                  gaussian_filter_smoothing_sigma=0.7,
                                  zoom_data_scale = 1., 
                                  fill_bottom_with_cmap_over_color=False,
                                  bottom_fill_bounds=None,
                                  add_shapes = {},
                                  round_xticks_to=1,
                                  round_yticks_to=0,
                                  inline_spacing=5.,
                                  include_top_bar=True,
                                  include_cbar=True,
                                  include_axis_labels=True,
                                  include_x_axis_ticklabels=True,
                                  include_y_axis_ticklabels=True,
                                  show_top_ticklabels=True,
                                  fig_ax_to_use=None, # only used when include_top_bar is False. If fig_ax_to_use is provided, images and gifs are not saved.
                                  ):
    
    
    results = np.array(w_data_vs_x_y_at_multiple_z)
    
    for i in range(len(units_opening_brackets)):
        if units_on_newline[i] and not ("\n" in units_opening_brackets[i]):
            units_opening_brackets[i].replace(" ", "")
            units_opening_brackets[i] = "\n" + units_opening_brackets[i]
            
    if zoom_data_scale>1:
        results2 = []
        for i in range(len(z_data)):
            results2.append(np.kron(results[i], np.ones((zoom_data_scale*len(x_data), zoom_data_scale*len(y_data)))))
        results = results2
        x_data = np.kron(x_data, np.ones(zoom_data_scale*len(x_data)))
        y_data = np.kron(y_data, np.ones(zoom_data_scale*len(y_data)))
        # results[np.isnan(results)] = 101010101011101
        # results = zoom(results, zoom_data_scale)
        # x_data = zoom(x_data, zoom_data_scale)
        # y_data = zoom(y_data, zoom_data_scale)
        # # z_data = zoom(z_data, zoom_data_scale)
        # results[results == 101010101011101] = np.nan
        
    if type(cmap)==str:
        cmap = mpl.colormaps[cmap]
    plt.rcParams['font.sans-serif'] = "Arial Unicode"
    plt.rcParams['font.size'] = str(default_fontsize)
    def create_frame(z_index):
        fig, axs, ax = None, None, None
        
        if include_top_bar:
            fig, axs = plt.subplots(2, 1, constrained_layout=True, 
                                    gridspec_kw=gridspec_kw)
            fig.set_figwidth(figwidth)
            ax = axs[0]
            a = [z_data[z_index]]
            ax.hlines(1,1,1)
            ax.set_xlim(min(z_ticks), max(z_ticks))
            ax.set_ylim(0.5,1.5)
            ax.xaxis.tick_top()
            if include_axis_labels:
                ax.set_xlabel(z_label + units_opening_brackets[2] + z_units + units_closing_brackets[2],  
                              fontsize=axis_title_fonts['size']['z'],
                              **fontname)
            ax.set_xticks(z_ticks,
                          **fontname)
        
            y = np.ones(np.shape(a))
            ax.plot(a,y,
                    color=z_marker_color, 
                    marker=z_marker_type,
                    ms = 7,)
            ax.axes.get_yaxis().set_visible(False)
            ax.tick_params(labelsize=axis_tick_fontsize)
            ax = axs[1]
        
        elif fig_ax_to_use is None:
            fig, axs = plt.subplots(1, 1, constrained_layout=True, 
                                    # gridspec_kw=gridspec_kw,
                                    )
            fig.set_figwidth(figwidth)
            axs = [axs]
            ax = axs[0]
        
        else:
            fig, ax = fig_ax_to_use
            if figwidth is not None: fig.set_figwidth(figwidth)
            
        if cmap_over_color is not None:
            cmap.set_over(cmap_over_color)
        if cmap_under_color is not None:
            cmap.set_under(cmap_under_color)
            
        results_data = results[z_index]

        
        ## Ensure contour labels remain inside plot area
        
        x_data_i, y_data_i = np.meshgrid(x_data, y_data)    
        results_data_i = results_data
        
        mask = np.ones(x_data_i.shape).astype(bool)
        mask = mask & (x_data_i>=x_ticks[0]) & (x_data_i<=x_ticks[-1])
        mask = mask & (y_data_i>=y_ticks[0]) & (y_data_i<=y_ticks[-1])
        
        x_data_i = np.ma.masked_where(~mask , x_data_i)
        y_data_i = np.ma.masked_where(~mask , y_data_i)
        results_data_i = np.ma.masked_where(~mask , results_data_i)
        
        if gaussian_filter_smoothing:
            results_data_i = gaussian_filter(results_data_i, gaussian_filter_smoothing_sigma)
        
        im = ax.contourf(x_data_i, y_data_i, results_data_i,
                          cmap=cmap,
                         levels=w_levels,
                         extend=extend_cmap
                         
                         )
        
        ## Ensure contour labels remain inside plot area
        ax.set_xlim((x_ticks[0], x_ticks[-1]))
        ax.set_ylim((y_ticks[0], y_ticks[-1]))
        ##
        
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        
        # Fill bottom with cmap_over_color
        if fill_bottom_with_cmap_over_color:
            if not bottom_fill_bounds:
                y1, y2, y3 = y_ticks[0], y_data[3], y_data[0]
                x1, x2, x3, x4 = (x_ticks[0], x_data[0], x_data[1], x_ticks[-1])
                ax.fill_between((x1, x2, x3, x4), (y1, y2, y3, y3), (y1, y1, y1, y1), color=cmap_over_color, zorder=0, lw=0)
            else:
                y1, y2, y3 = bottom_fill_bounds[0][1], bottom_fill_bounds[1][1], bottom_fill_bounds[2][1]
                x1, x2, x3, x4 = bottom_fill_bounds[0][0], bottom_fill_bounds[1][0], bottom_fill_bounds[1][0], bottom_fill_bounds[2][0]
                ax.fill_between((x1, x2, x3, x4), (y1, y2, y3, y3), (y1, y1, y1, y1), color=cmap_over_color, zorder=0, lw=0)
        # ########--########
        ax.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            direction='inout',
            right=True,
            width=0.65,
            labelsize=axis_tick_fontsize,
            # zorder=200,
            )

        ax.tick_params(
            axis='y',          
            which='major',      
            length=7,
            )

        ax.tick_params(
            axis='y',          
            which='minor',      
            length=3.5,
            )
        
        
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            direction='inout',
            right=True,
            top=True,
            width=0.65,
            labelsize=axis_tick_fontsize,
            # zorder=200,
            )
        ax.tick_params(
            axis='x',          
            which='major',      
            length=7,
            # right=True,
            # top=True,
            )

        ax.tick_params(
            axis='x',          
            which='minor',      
            length=3.5,
            # right=True,
            # top=True,
            )
        
        # ax2 = ax.twinx()
        
        
        # if not (y_ticks==[] or x_ticks==[]):
        #     ax.set_yticks(y_ticks)
        #     ax2.set_yticks(y_ticks)
        #     l = ax.get_ylim()
        #     l2 = ax2.get_ylim()
        #     f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        #     ticks = f(ax.get_yticks())
        #     ax2.yaxis.set_major_locator(FixedLocator(ticks))
        #     ax2.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        
        # else:
        #     ax2.set_yticks(ax.get_y_ticks())
        #     l = ax.get_ylim()
        #     l2 = ax2.get_ylim()
        #     f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        #     ticks = f(ax.get_yticks())
        #     ax2.yaxis.set_major_locator(FixedLocator(ticks))
        #     ax2.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
            
        #     ax2.set_yticks(ax.get_x_ticks())
        #     l = ax.get_xlim()
        #     l2 = ax2.get_xlim()
        #     f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        #     ticks = f(ax.get_xticks())
        #     ax2.xaxis.set_major_locator(FixedLocator(ticks))
        #     ax2.xaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
            
        # ax2.tick_params(
        #     axis='y',          
        #     which='both',      
        #     direction='in',
        #     # right=True,
        #     labelright=False,
        #     width=1,
        #     )
        
        # ########--########
        
        if not list(comparison_range)==[]:
            [m1,n1] = np.where((results_data_i > comparison_range[0]) & (results_data_i < comparison_range[1]))
            # [m2,n2] = np.where(results_data_i < 7.5)
            
            z1 = np.zeros(results_data_i.shape)
            z1[m1,n1] = 99
            
            # print(z1,)
            plt.rcParams['hatch.linewidth'] = 0.6
            plt.rcParams['hatch.color'] = 'white'
            cs = ax.contourf(x_data_i, y_data_i, z1 ,1 , hatches=['', comparison_range_hatch_pattern],  alpha=0.,
                             # extend=extend_cmap,
                             )
        
            
        # clines = ax.contour(x_data_i, y_data_i, results_data_i,
        #            levels=w_ticks,
        #             colors='black',
        #             # colors=None,
        #            linewidths=w_tick_width)
        
        # clabs = ax.clabel(clines, 
        #            w_ticks,
        #            fmt=fmt_clabel, 
        #           fontsize=clabel_fontsize,
        #           colors='black',
        #           inline_spacing=inline_spacing,
        #           )
        manual_clabels_regular_keys = list(manual_clabels_regular.keys())
        nonmanual_ticks_levels = [i for i in w_ticks if not i in manual_clabels_regular_keys]
        
        if manual_clabels_regular:

            #redraw relevant lines
            clines2 = ax.contour(x_data_i, y_data_i, results_data_i,
                       levels=manual_clabels_regular_keys,
                        colors='black',
                       linewidths=w_tick_width)
            
            ## draw inline labels over both sets of lines
            ax.clabel(clines2, 
                       manual_clabels_regular_keys,
                       fmt=fmt_clabel, 
                      fontsize=clabel_fontsize,
                      colors='black',
                      inline=True,
                      manual=[manual_clabels_regular[i] for i in manual_clabels_regular_keys],
                      inline_spacing=inline_spacing,
                      )
        
            
            # nonmanual_ticks_levels = [i for i in w_ticks if not i in manual_clabels_regular_keys]
            # if not len(w_ticks) == len(manual_clabels_regular_keys):
            #     clines = ax.contour(x_data_i, y_data_i, results_data_i,
            #                levels=nonmanual_ticks_levels,
            #                 colors='black',
            #                 # colors=None,
            #                linewidths=w_tick_width)
                
            #     # clabs = ax.clabel(clines, 
            #     #            nonmanual_ticks_levels,
            #     #            fmt=fmt_clabel, 
            #     #           fontsize=clabel_fontsize,
            #     #           colors='black',
            #     #           inline_spacing=inline_spacing,
            #     #           )
        

        
        # else:
            
        # automatic lines and labels
        clines = ax.contour(x_data_i, y_data_i, results_data_i,
                   levels=nonmanual_ticks_levels,
                    colors='black',
                    # colors=None,
                   linewidths=w_tick_width)
        
        try:
            clabs = ax.clabel(clines, 
                       nonmanual_ticks_levels,
                       fmt=fmt_clabel, 
                      fontsize=clabel_fontsize,
                      colors='black',
                      inline_spacing=inline_spacing,
                      )
        except:
            pass
        
        if label_over_color:
            nonmanual_ticks_levels.remove(w_ticks[-1])
            location_from_auto_labeling = (clabs[-1]._x, clabs[-1]._y)
            #redraw relevant lines
            clines2 = ax.contour(x_data_i, y_data_i, results_data_i,
                       levels=[w_ticks[-1]],
                        colors='black',
                       linewidths=w_tick_width)
            ## draw inline labels over both sets of lines
            ax.clabel(clines2, 
                       [w_ticks[-1]],
                       fmt=fmt_clabel, 
                      fontsize=clabel_fontsize,
                      colors=label_over_color,
                      inline=True,
                      manual=[location_from_auto_labeling],
                      inline_spacing=inline_spacing,
                      )
        if label_under_color:
            nonmanual_ticks_levels.remove(w_ticks[0])
            location_from_auto_labeling = (clabs[0]._x, clabs[0]._y)
            #redraw relevant lines
            clines2 = ax.contour(x_data_i, y_data_i, results_data_i,
                       levels=[w_ticks[0]],
                        colors='black',
                       linewidths=w_tick_width)
            ## draw inline labels over both sets of lines
            ax.clabel(clines2, 
                       [w_ticks[0]],
                       fmt=fmt_clabel, 
                      fontsize=clabel_fontsize,
                      colors=label_under_color,
                      inline=True,
                      manual=[location_from_auto_labeling],
                      inline_spacing=inline_spacing,
                      )
        try:
            if not list(comparison_range)==[]:
                clines3 = ax.contour(x_data_i, y_data_i, results_data_i,
                           levels=comparison_range,
                            colors='white',
                           linewidths=w_tick_width,
                           # zorder=199,
                           )
                
                if manual_clabels_comparison_range:
                    ax.clabel(clines3, 
                               comparison_range,
                               fmt=fmt_clabel, 
                              fontsize=clabel_fontsize,
                              colors='black',
                              manual=[manual_clabels_comparison_range[i] for i in comparison_range],
                              inline_spacing=inline_spacing,
                              )
                else:
                    ax.clabel(clines3, 
                               comparison_range,
                               fmt=fmt_clabel, 
                              fontsize=clabel_fontsize,
                              colors='black',
                              inline_spacing=inline_spacing,
                              )
        except:
            pass
        
        if include_axis_labels:
            ax.set_ylabel(y_label + units_opening_brackets[1] + y_units + units_closing_brackets[1],  
                          fontsize=axis_title_fonts['size']['y'],
                          **fontname)
            ax.set_xlabel(x_label + units_opening_brackets[0] + x_units + units_closing_brackets[0], 
                          fontsize=axis_title_fonts['size']['x'],
                          **fontname)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        plt.rcParams["axes.axisbelow"] = False
        
        if not contourplot_facecolor.all()==None:
            ax.set_facecolor(contourplot_facecolor)
        
        if text_boxes:
            text_boxes_keys = list(text_boxes.keys())
            for i in text_boxes_keys:
                (xpos, ypos), textcolor = text_boxes[i]
                ax.text(xpos, ypos, i, color=textcolor, fontsize=clabel_fontsize)
        
        if additional_vlines:
            ax.vlines(additional_vlines, y_ticks[0], y_ticks[-1], linewidth=1, 
                      color=additional_vline_colors,
                      linestyles=additional_vline_linestyles,
                      linewidths=additional_vline_linewidths)
        if additional_hlines:
            ax.hlines(additional_hlines, x_ticks[0], x_ticks[-1], linewidth=1, 
                      color=additional_hline_colors,
                      linestyles=additional_hline_linestyles,
                      linewidths=additional_hline_linewidths)
        
        if additional_points:
            additional_point_keys = additional_points.keys()
            for apk in additional_point_keys:
                xp, yp = apk
                markershape, markercolor, markersize = additional_points[apk]
                ax.plot(xp, yp, c='k', 
                        # label='.',
                        marker=markershape, 
                            markersize=markersize, 
                            markerfacecolor=markercolor,
                              markeredgewidth=0.8,
                             zorder=500)
        
        if not add_shapes=={}:
            for coords, (shapecolor, shapezorder) in  add_shapes.items():
                t1 = plt.Polygon(coords, color=shapecolor, zorder=shapezorder)
                ax.add_patch(t1,)
                
        if not list(comparison_lines)==[]:
            
            cs = ax.contour(x_data_i, y_data_i, results_data_i,
                       levels=comparison_lines,
                        colors=comparison_lines_colors,
                       linewidths=w_tick_width)
            ax.clabel(cs, 
                       comparison_lines,
                       fmt=fmt_clabel, 
                      fontsize=clabel_fontsize,
                      colors=comparison_lines_colors,
                      inline_spacing=inline_spacing,
                      )
                      
        
        norm = mpl.colors.BoundaryNorm(w_levels, cmap.N, extend=extend_cmap)
        
        # if not cbar_ticks:
        #     cbar_ticks = w_levels
        
        if include_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(
                                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                                # im,
                                cax=cax, 
                         ticks = cbar_ticks)
            
            cbar.set_label(label=w_label + units_opening_brackets[3] + w_units + units_closing_brackets[3], 
                                                  size=axis_title_fonts['size']['w'],
                                                  loc='center',
                                                  **fontname
                                                  )
            
            
            # cbar.ax.set_minor_locator(AutoMinorLocator(cbar_n_minor_ticks+1))
            cbar.ax.minorticks_on()
            
            # set minor ticks
            curr_major_ticks = [float(i) for i in cbar.get_ticks(minor=False)]
            n_major_ticks = len(curr_major_ticks)
            major_tick_step_size = curr_major_ticks[1] - curr_major_ticks[0]
            minor_tick_step_size = major_tick_step_size/(1+cbar_n_minor_ticks)
            curr_tick =  curr_major_ticks[0]
            curr_minor_ticks = []
            while curr_tick<curr_major_ticks[-1]:
                curr_tick+=minor_tick_step_size
                if curr_tick not in curr_major_ticks:
                    curr_minor_ticks.append(curr_tick)
            cbar.set_ticks(curr_minor_ticks, minor=True)
            #
            
            cbar.ax.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                direction='inout',
                # right=True,
                width=0.65,
                labelsize=axis_tick_fontsize,
                )
            cbar.ax.tick_params(
                axis='x',          
                which='major',      
                length=7,
                )
    
            cbar.ax.tick_params(
                axis='x',          
                which='minor',      
                length=3.5,
                )
            # plt.rcParams['hatch.linewidth'] = 0.6
            # plt.rcParams['hatch.color'] = 'white'
            cbar.ax.fill_betweenx(comparison_range,
                                  # cbar.ax.get_xlim()[0],cbar.ax.get_xlim()[1],
                                  -1, 2,
                               facecolor='none', 
                               hatch=comparison_range_hatch_pattern,
                                # zorder=200,
                                linewidth=0.6,
                                edgecolor='white',
                               # alpha=0.,
                               )
        
        ax.set_title(' ', fontsize=gap_between_figures)

        ax.set_axisbelow(False)
        
        def round_to(val, round_to):
            if isinstance(val, str):
                return val
            else:
                if round_to<1:
                    return int(np.round(val,round_to))
                else:
                    return np.round(val,round_to)
                
                        
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{round_to(val,round_xticks_to)}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{round_to(val,round_yticks_to)}'))
            
        if not show_top_ticklabels: # !!! Update rounding methods
            xticks_new = [round_to(i,round_xticks_to) for i in x_ticks.copy()]
            xticks_new[-1] = ''
            ax.set_xticklabels(xticks_new)
            yticks_new = [round_to(i,round_yticks_to) for i in y_ticks.copy()]
            yticks_new[-1] = ''
            ax.set_yticklabels(yticks_new)
            
        if not include_x_axis_ticklabels:
            ax.set_xticklabels([])
            
        if not include_y_axis_ticklabels:
            ax.set_yticklabels([])


        if fig_ax_to_use is None:
            plt.savefig(f'./{animated_contourplot_filename}_frame_{z_index}.png', 
                        transparent = False,  
                        facecolor = 'white',
                        bbox_inches='tight',
                        dpi=dpi,
                        )                                
            plt.close()
        return fig, axs
        
    
    fig_list, axs_list = [], []
    for z_index in range(len(z_data)):
        fig, axs = create_frame(z_index)
        fig_list.append(fig)
        axs_list.append(axs)
    
    if fig_ax_to_use is None:
        frames = []
        for z_index in range(len(z_data)):
            image = imageio.v2.imread(f'./{animated_contourplot_filename}_frame_{z_index}.png')
            frames.append(image)
        
        
        if keep_gifs:
            if n_loops==('inf' or 'infinite' or 'infinity' or np.inf):
                imageio.mimsave('./' + animated_contourplot_filename + '.gif',
                                frames,
                                fps=fps,
                                ) 
                frames.reverse()
                imageio.mimsave('./' + 'reverse_'+animated_contourplot_filename + '.gif',
                                frames,
                                fps=fps,
                                ) 
            else:
                imageio.mimsave('./' + animated_contourplot_filename + '.gif',
                                frames,
                                fps=fps,
                                loop=n_loops,
                                ) 
                frames.reverse()
                imageio.mimsave('./' + 'reverse_'+animated_contourplot_filename + '.gif',
                                frames,
                                fps=fps,
                                loop=n_loops,
                                ) 
        
        if not keep_frames:
            for z_index in range(len(z_data)):
                os.remove(f'./{animated_contourplot_filename}_frame_{z_index}.png')
    
    return fig_list, axs_list

#%% Animated barplot

def animated_barplot(y_data, # shape = z * x
                                  x_data,
                                  z_data,
                                  x_label="x", # title of the x axis
                                  z_label="z", # title of the z axis
                                  y_label="w", # title of the color axis
                                  x_ticks=[],
                                  z_ticks=[],
                                  y_ticks=[], # labeled, lined contours (a subset of w_levels)
                                  x_units="",
                                  z_units="",
                                  y_units="",
                                  gridspec_kw={'height_ratios': [1, 20]},
                                  fontname={'fontname':'Arial Unicode'},
                                  figwidth=3.9,
                                  bar_width = 0.5,
                                  dpi=600,
                                  z_marker_color='b',
                                  z_marker_type='v',
                                  axis_title_fonts={'size': {'x': 12, 'y':12, 'z':12},},
                                  gap_between_figures=20., 
                                  fps=5, # animation frames (z values traversed) per second
                                  n_loops='inf', # the number of times the animated contourplot should loop animation over z; infinite by default
                                  animated_barplot_filename='animated_barplot',
                                  keep_frames=False, # leaves frame PNG files undeleted after running; False by default
                                  n_minor_ticks = 1,
                                  cbar_n_minor_ticks = 4,
                                  
                                  colors='blue',
                                  edgecolors='black',
                                  linewidths=0.65,
                                  
                                  default_fontsize=12.,
                                  units_on_newline = (True, False, False), # x,y,z,w
                                  contourplot_facecolor=np.array([None, None, None]),
                                  text_boxes = {}, # str: (x,y)
                                  additional_points = {}, # (x,y): (markershape, markercolor, markersize)
                                  additional_vlines = [],
                                  additional_vline_colors='black',
                                  additional_vline_linestyles='dashed',
                                  additional_vline_linewidths=0.8,
                                  additional_hlines = [],
                                  additional_hline_colors='black',
                                  additional_hline_linestyles='dashed',
                                  additional_hline_linewidths=0.8,
                                  axis_tick_fontsize=12.,
                                  gaussian_filter_smoothing=False,
                                  gaussian_filter_smoothing_sigma=0.7,
                                  ):
    
    
    results = np.array(y_data)
    

    plt.rcParams['font.sans-serif'] = "Arial Unicode"
    plt.rcParams['font.size'] = str(default_fontsize)
    def create_frame(z_index):
        fig, axs = plt.subplots(2, 1, constrained_layout=True, 
                                gridspec_kw=gridspec_kw)
        fig.set_figwidth(figwidth)
        ax = axs[0]
        a = [z_data[z_index]]
        ax.hlines(1,1,1)
        
        if list(z_ticks): 
            ax.set_xlim(min(z_ticks), max(z_ticks))
        else:
            ax.set_xlim(z_data.min(), z_data.max())
            
        ax.set_ylim(0.5,1.5)
        
        
        
        ax.xaxis.tick_top()
        units_opening_brackets = [" [", " [", " [", " ["]
        for i in range(len(units_opening_brackets)):
            if units_on_newline:
                units_opening_brackets[i] = "\n["
                
        ax.set_xlabel(z_label + units_opening_brackets[2] + z_units + "]",  
                      fontsize=axis_title_fonts['size']['z'],
                      **fontname)
        ax.set_xticks(z_ticks,
                      **fontname)
    
        y = np.ones(np.shape(a))
        ax.plot(a,y,
                color=z_marker_color, 
                marker=z_marker_type,
                ms = 7,)
        ax.axes.get_yaxis().set_visible(False)
        ax.tick_params(labelsize=axis_tick_fontsize)
        ax = axs[1]

        if list(y_ticks): 
            ax.set_ylim(min(y_ticks), max(y_ticks))
        else:
            ax.set_ylim(y_data.min(), y_data.max())
    
        # results_data = results[z_index]

        ax.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        
        ax.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            direction='inout',
            right=True,
            width=0.65,
            labelsize=axis_tick_fontsize,
            # zorder=200,
            )

        ax.tick_params(
            axis='y',          
            which='major',      
            length=7,
            )

        ax.tick_params(
            axis='y',          
            which='minor',      
            length=3.5,
            )
        
        
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            direction='inout',
            right=True,
            top=True,
            width=0.65,
            labelsize=axis_tick_fontsize,
            # zorder=200,
            )
        ax.tick_params(
            axis='x',          
            which='major',      
            length=7,
            # right=True,
            # top=True,
            )

        ax.tick_params(
            axis='x',          
            which='minor',      
            length=3.5,
            # right=True,
            # top=True,
            )
        
       
        ax.bar([i for i in range(len(x_data))], 
                   y_data[z_index],
                   width=bar_width,
                   color=colors,
                   edgecolor=edgecolors,
                   linewidth=linewidths,
                   )
        
        
        ax.set_xticks(x_ticks,
                      **fontname)
        ax.set_yticks(y_ticks,
                      **fontname)
        
        ax.set_title(' ', fontsize=gap_between_figures)
        
        ax.set_ylabel(y_label + " [" + y_units + "]", 
                      fontname, fontsize=axis_title_fonts['size']['y'], 
                   # fontweight='bold',
                   )
        
        plt.savefig(f'./{animated_barplot_filename}_frame_{z_index}.png', 
                    transparent = False,  
                    facecolor = 'white',
                    bbox_inches='tight',
                    dpi=dpi,
                    )                                
        plt.close()
        
        
    for z_index in range(len(z_data)):
        create_frame(z_index)
          
    frames = []
    for z_index in range(len(z_data)):
        image = imageio.v2.imread(f'./{animated_barplot_filename}_frame_{z_index}.png')
        frames.append(image)
    
    
    if n_loops==('inf' or 'infinite' or 'infinity' or np.inf):
        imageio.mimsave('./' + animated_barplot_filename + '.gif',
                        frames,
                        fps=fps,
                        ) 
    else:
        imageio.mimsave('./' + animated_barplot_filename + '.gif',
                        frames,
                        fps=fps,
                        loop=n_loops,
                        ) 
    
    if not keep_frames:
        for z_index in range(len(z_data)):
            os.remove(f'./{animated_barplot_filename}_frame_{z_index}.png')

#%% Animated stacked barplot

def animated_stacked_barplot(   dataframes_over_z, 
                               z_data,
                               y_ticks=[], x_ticks=[], 
                               ylim=[],
                               z_label="z", # title of the z axis
                               z_ticks=[],
                               z_units="",
                               hatch_patterns=('\\', '//', '|', 'x',),
                               colormap=None,
                               metric_total_values=[], metric_units=[],
                               y_label='', y_units='',
                               linewidth=0.8,
                               filename='stacked_bar_plot',
                               dpi=600,
                               # fig_width=7,
                               figheight=5.5*1.1777,
                               show_totals=False,
                               totals=[],
                               sig_figs_for_totals=3,
                               units_list=[],
                               totals_label_text=r"$\bfsum:$",
           
                                  gridspec_kw={'height_ratios': [1, 20]},
                                  fontname={'fontname':'Arial Unicode'},
                                  figwidth=3.9,
                                  bar_width = 0.5,
                                  z_marker_color='b',
                                  z_marker_type='v',
                                  axis_title_fonts={'size': {'x': 12, 'y':12, 'z':12},},
                                  gap_between_figures=20., 
                                  fps=5, # animation frames (z values traversed) per second
                                  n_loops='inf', # the number of times the animated contourplot should loop animation over z; infinite by default
                                  animated_barplot_filename='animated_barplot',
                                  keep_frames=False, # leaves frame PNG files undeleted after running; False by default
                                  n_minor_ticks = 1,
                                  cbar_n_minor_ticks = 4,
                                  
                                  colors='blue',
                                  edgecolors='black',
                                  linewidths=0.65,
                                  
                                  default_fontsize=12.,
                                  units_on_newline = (True, False, False), # x,y,z,w
                                  contourplot_facecolor=np.array([None, None, None]),
                                  text_boxes = {}, # str: (x,y)
                                  additional_points = {}, # (x,y): (markershape, markercolor, markersize)
                                  additional_vlines = [],
                                  additional_vline_colors='black',
                                  additional_vline_linestyles='dashed',
                                  additional_vline_linewidths=0.8,
                                  additional_hlines = [],
                                  additional_hline_colors='black',
                                  additional_hline_linestyles='dashed',
                                  additional_hline_linewidths=0.8,
                                  axis_tick_fontsize=12.,
                                  ):
    
    

    plt.rcParams['font.sans-serif'] = "Arial Unicode"
    plt.rcParams['font.size'] = str(default_fontsize)
    def create_frame(z_index):
        fig, axs = plt.subplots(2, 1, constrained_layout=True, 
                                gridspec_kw=gridspec_kw)
        fig.set_figwidth(figwidth)
        fig.set_figheight(figheight)
        ax = axs[0]
        a = [z_data[z_index]]
        ax.hlines(1,1,1)
        
        if list(z_ticks): 
            ax.set_xlim(min(z_ticks), max(z_ticks))
        else:
            ax.set_xlim(z_data.min(), z_data.max())
            
        ax.set_ylim(0.5,1.5)
        
        
        
        ax.xaxis.tick_top()
        units_opening_brackets = [" [", " [", " [", " ["]
        for i in range(len(units_opening_brackets)):
            if units_on_newline:
                units_opening_brackets[i] = "\n["
                
        ax.set_xlabel(z_label + units_opening_brackets[2] + z_units + "]",  
                      fontsize=axis_title_fonts['size']['z'],
                      **fontname)
        ax.set_xticks(z_ticks,
                      **fontname)
    
        y = np.ones(np.shape(a))
        ax.plot(a,y,
                color=z_marker_color, 
                marker=z_marker_type,
                ms = 7,)
        ax.axes.get_yaxis().set_visible(False)
        ax.tick_params(labelsize=axis_tick_fontsize)
        ax = axs[1]
        
        # Stacked bar plot
        dataframe = dataframes_over_z[z_index]
        # dataframe = dataframe_over_z.loc[dataframe_over_z['Time'] == z_data[z_index]]
        
        ax = dataframe.T.plot(kind='bar', stacked=True, edgecolor='k', linewidth=linewidth,
                              color=colors,
                              colormap=colormap,
                              # facecolor="white",
                               # use_index=False,
                              rot=0,
                              )
        
        
        ax.set_facecolor("white")
        
        fig = plt.gcf()
        
        
        
        ax.set_yticks(y_ticks,)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val}'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        
        
        ax.containers[:4]
        # print(ax.containers)
        bars = [thing for thing in ax.containers if isinstance(thing, BarContainer)]
        
        
        used_facecolors = []
        used_hatches_dict = {}
        # print(len(bars))
        bar_hatch_dict = {}
        bar_num = 0
        patterns = itertools.cycle(hatch_patterns)
        for bar in bars:
            # print(len(bar))
            for patch in bar:
                
                if not bar_num in bar_hatch_dict.keys():
                    curr_facecolor = patch.get_facecolor()
                    
                    if not curr_facecolor in used_hatches_dict.keys():
                        used_hatches_dict[curr_facecolor] = []
                        
                    if curr_facecolor in used_facecolors:
                        used_hatches = used_hatches_dict[curr_facecolor]
                        curr_hatch = next(patterns)
                        i=100
                        while curr_hatch in used_hatches:
                            i+=1
                            # print(True)
                            curr_hatch = next(patterns)
                            if i>100:
                                break
                        patch.set_hatch(curr_hatch)
                        bar_hatch_dict[bar_num] = curr_hatch
                        used_hatches.append(curr_hatch)
                    else:
                        bar_hatch_dict[bar_num] = None
                    used_facecolors.append(curr_facecolor)
                else:
                    if bar_hatch_dict[bar_num] is not None:
                        if curr_facecolor in used_facecolors:
                            patch.set_hatch(bar_hatch_dict[bar_num])
            bar_num+=1
        # print(bar_hatch_dict)
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', edgecolor='white')

        ax.set_ylabel(y_label + " [" + y_units + "]", fontsize=14)
        
        # ax.set_xlabel(x_labels, fontsize=14)
        
        wrap_labels(ax,10)
        
        ax.axhline(y=0,  color='k', linestyle='-', linewidth=linewidth)
        
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            direction='inout',
            # right=True,
            width=1,
            )

        ax.tick_params(
            axis='y',          
            which='major',      
            length=5,
            )

        ax.tick_params(
            axis='y',          
            which='minor',      
            length=3,
            )
        
        
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            direction='inout',
            # right=True,
            width=1,
            )
        
        
        ax2 = ax.twinx()
        
        
        if not list(y_ticks)==[]:
            ax.set_yticks(y_ticks)
            ax2.set_yticks(y_ticks)
            l = ax.get_ylim()
            l2 = ax2.get_ylim()
            f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
            ticks = f(ax.get_yticks())
            ax2.yaxis.set_major_locator(FixedLocator(ticks))
            ax2.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        
        else:
            ax2.set_yticks(ax.get_y_ticks())
            l = ax.get_ylim()
            l2 = ax2.get_ylim()
            f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
            ticks = f(ax.get_yticks())
            ax2.yaxis.set_major_locator(FixedLocator(ticks))
            ax2.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
            
        ax2.tick_params(
            axis='y',          
            which='both',      
            direction='in',
            # right=True,
            labelright=False,
            width=1,
            )
        
        if ylim: 
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([min(y_ticks), max(y_ticks)])
        # plt.tight_layout()
        
        if show_totals:
            
            num_x_points = len(dataframe.columns)
            distance_between_x_points = 1./num_x_points
            start_x_coord = distance_between_x_points/2.
            
            ax.annotate(
                    xy=(start_x_coord-0.75*distance_between_x_points, 
                    1.1), 
                    text=totals_label_text, 
                    # fontsize=14, 
                    ha='center', va='center',
                    xycoords='axes fraction',
                     # transform=plt.gcf().transFigure,
                     )
            
            
            for i in range(num_x_points):
                ax.annotate(
                        xy=(start_x_coord+i*distance_between_x_points, 
                        1.1), 
                        text=get_rounded_str(totals[i], sig_figs_for_totals), 
                        # fontsize=14, 
                        ha='center', va='center',
                        xycoords='axes fraction',
                         # transform=plt.gcf().transFigure,
                         )
                ax.annotate(
                        xy=(start_x_coord+i*distance_between_x_points, 
                        1.05), 
                        text=units_list[i], 
                        # fontsize=14, 
                        ha='center', va='center',
                        xycoords='axes fraction',
                         # transform=plt.gcf().transFigure,
                         )
                
        # plt.savefig(filename+'.png', dpi=dpi, bbox_inches='tight',
        #             facecolor=fig.get_facecolor(),
        #             transparent=False)
        
        # plt.show()
        
        ax.set_title(' ', fontsize=gap_between_figures)
        
        ax.set_ylabel(y_label + " [" + y_units + "]", 
                      fontname, fontsize=axis_title_fonts['size']['y'], 
                   # fontweight='bold',
                   )
        
        plt.savefig(f'./{animated_barplot_filename}_frame_{z_index}.png', 
                    transparent = False,  
                    facecolor = 'white',
                    bbox_inches='tight',
                    dpi=dpi,
                    )                                
        plt.close()
        
        
    for z_index in range(len(z_data)):
        create_frame(z_index)
          
    frames = []
    for z_index in range(len(z_data)):
        image = imageio.v2.imread(f'./{animated_barplot_filename}_frame_{z_index}.png')
        frames.append(image)
    
    
    if n_loops==('inf' or 'infinite' or 'infinity' or np.inf):
        imageio.mimsave('./' + animated_barplot_filename + '.gif',
                        frames,
                        fps=fps,
                        ) 
    else:
        imageio.mimsave('./' + animated_barplot_filename + '.gif',
                        frames,
                        fps=fps,
                        loop=n_loops,
                        ) 
    
    if not keep_frames:
        for z_index in range(len(z_data)):
            os.remove(f'./{animated_barplot_filename}_frame_{z_index}.png')
            
            
#%%
def box_and_whiskers_plot(uncertainty_data, # either an iterable of uncertainty data (for a single boxplot) or a nested list of iterables of uncertainty data (for multiple plots))
                          baseline_values=None, 
                          baseline_locations=[1,], # any number in range 1 - # of boxes (inclusive)
                          baseline_marker_shapes=['D', 'D', 'D'],
                          baseline_marker_colors=['w'],
                          baseline_marker_sizes=[6,],
                          ranges_for_comparison=None, # [(v1, v2), (x1, x2), ...]
                          ranges_for_comparison_colors=["#c0c1c2",], # [c1, c2, ...]
                          values_for_comparison=[],
                          n_minor_ticks=1,
                          y_label='Metric',
                          y_units='units',
                          y_ticks=[],
                          show_x_ticks=False,
                          x_tick_labels=None,
                          x_tick_wrap_width=6,
                          boxcolor="#A97802",
                          save_file=True,
                          filename='box_and_whiskers_plot',
                          dpi=600,
                          fig_width=1.5,
                          fig_height=5.5,
                          box_width=1.5,
                          height_ratios = [1],
                          width_ratios = [1,5],
                          xlabelpad=5,
                          ylabelpad=5,
                          xticks_fontsize = 17,
                          ylabel_fontsize = 19,
                          yticks_fontsize = 17,
                          default_fontsize = 15,
                          show=False,
                          n_cols_subplots=1,
                          xticks_fontcolor='black',
                          rotate_xticks=0.,
                          background_fill_colors=None,
                          background_fill_alphas=None,
                          ):
    n_boxes = 1. if not hasattr(uncertainty_data[0], '__iter__') else len(uncertainty_data)
    plt.rcParams['font.sans-serif'] = "Arial Unicode"
    plt.rcParams['font.size'] = str(default_fontsize)

    gridspec_kw={
                'height_ratios': height_ratios,
                 'width_ratios': width_ratios,
                 }
    
    fig, axs = plt.subplots(1,n_cols_subplots, gridspec_kw=gridspec_kw, constrained_layout = True) if n_cols_subplots>1\
        else plt.subplots(1,1,constrained_layout = True)
    
    ax = axs[0] if n_cols_subplots>1 else axs
    # ax = plt.subplot(1, 2, 1,
    #                         # constrained_layout=True, 
    #                         # gridspec_kw=gridspec_kw,
    #                         )
    
    fig = plt.gcf()
    
    if n_boxes > 1.:
        xrange = [i+0.5 for i in range(n_boxes+1)]
        ax.set_xlim(min(xrange), max(xrange))
        baseline_marker_sizes = [6 for i in range(n_boxes)]
        baseline_marker_colors = ['w' for i in range(n_boxes)]
    # ax.set_facecolor("white")
    
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)


    ax.boxplot(x=uncertainty_data, patch_artist=True,
                        widths=box_width, whis=[5, 95], vert=True, flierprops = {'marker': ''},
                        boxprops={'facecolor': boxcolor,
                                   'edgecolor':'black',
                                   'linewidth':0.8},
                         medianprops={'color':'black',
                                      'linewidth':0.8},
                         whiskerprops={'linewidth':0.8},
                         zorder=99)
    
    if baseline_values is not None:
        for i in range(len(baseline_locations)):
            baseline_val, baseline_loc, baseline_marker_color, baseline_marker_size =\
                baseline_values[i], baseline_locations[i], baseline_marker_colors[i], baseline_marker_sizes[i]
            baseline_marker_shape = baseline_marker_shapes[i]
            ax.plot(baseline_loc, baseline_val, c='k', 
                    # label='.',
                    marker=baseline_marker_shape, 
                        markersize=baseline_marker_size, 
                        markerfacecolor=baseline_marker_color,
                          markeredgewidth=0.8,
                         zorder=100)

    if not (ranges_for_comparison==() or ranges_for_comparison==[] or ranges_for_comparison==None):
        for i in range(len(ranges_for_comparison)):
            range_for_comparison, range_for_comparison_color = ranges_for_comparison[i], ranges_for_comparison_colors[i]
            ax.fill_between(list(range(0, n_boxes+2)), [range_for_comparison[0]], [range_for_comparison[1]], color=range_for_comparison_color, alpha=1., zorder=1)

    if not (values_for_comparison==() or values_for_comparison==[] or values_for_comparison==None):
        1 # not implemented
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
    
    if not show_x_ticks:
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
    else:
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            direction='inout',
            length=5,
            width=1,
            labelbottom=False if x_tick_labels is None else True,
            zorder=5,)
    
    if show_x_ticks and (x_tick_labels is not None):
        ax.set_xticks(ticks=list(range(1,n_boxes+1)), 
                      labels=x_tick_labels, 
                      fontsize=xticks_fontsize,
                      color=xticks_fontcolor)
        wrap_labels(ax, width=x_tick_wrap_width, fontsize=xticks_fontsize)
    

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        direction='inout',
        # right=True,
        width=1,
        labelsize=yticks_fontsize,
        zorder=5,
        )

    ax.tick_params(
        axis='y',          
        which='major',      
        length=5,
        zorder=5,
        )

    ax.tick_params(
        axis='y',          
        which='minor',      
        length=3,
        zorder=5,
        )

    ax2 = ax.twinx()
    
    ax.set_ylim(min(y_ticks), max(y_ticks))
    ax2.set_ylim(min(y_ticks), max(y_ticks))
    
    if not list(y_ticks)==[]:
        ax.set_yticks(y_ticks)
        ax2.set_yticks(y_ticks)
        l = ax.get_ylim()
        l2 = ax2.get_ylim()
        f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        ticks = f(ax.get_yticks())
        ax2.yaxis.set_major_locator(FixedLocator(ticks))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
    
    else:
        ax2.set_yticks(ax.get_y_ticks())
        l = ax.get_ylim()
        l2 = ax2.get_ylim()
        f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        ticks = f(ax.get_yticks())
        ax2.yaxis.set_major_locator(FixedLocator(ticks))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        
        # loc = LinearLocator(numticks = len(y_ticks))
        # ax2.set_yticks(y_ticks)
        # ax.yaxis.set_major_locator(loc)
        # ax2.yaxis.set_major_locator(loc)
        # nticks = len(y_ticks)
        # ax.yaxis.set_major_locator(LinearLocator(nticks))
        # ax2.yaxis.set_major_locator(loc)
        
        # ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
        
    ax2.tick_params(
        axis='y',          
        which='both',      
        direction='in',
        # right=True,
        labelright=False,
        width=1,
        labelsize=yticks_fontsize,
        )
    
    ax.set_ylabel(y_label + " [" + y_units + "]", 
                  {'fontname':'Arial Unicode'}, fontsize=ylabel_fontsize, 
                  labelpad=ylabelpad,
               # fontweight='bold',
               )
    ax.xaxis.labelpad = xlabelpad
    ax.tick_params(axis='x', which='major', pad=xlabelpad)
    
    for label in ax.get_xticklabels():
        label.set_rotation(rotate_xticks)
        
    # ax.set_ylim(min(y_ticks), max(y_ticks))

    if background_fill_colors is not None:
        if background_fill_alphas is None: background_fill_alphas = [1]*len(background_fill_colors)
        # ax.fill_betweenx(y_ticks, -1, 0, 
        #                  color=background_fill_colors[0], 
        #                  alpha=background_fill_alphas[0],
        #                  zorder=2,)
        curr_xtick = 0.5
        for bg_col, bg_alpha in zip(
                                 # range(1, len(background_fill_colors)+1), 
                                 # np.linspace(0.5, len(background_fill_colors)+1, len(background_fill_colors)),
                                 background_fill_colors,
                                 background_fill_alphas):
            ax.fill_betweenx(y_ticks, 
                             curr_xtick, 
                             curr_xtick+1,
                             color=bg_col, alpha=bg_alpha,
                             zorder=2,
                             edgecolor="none", linewidth=0.0)
            curr_xtick += 1
        # ax.fill_betweenx(y_ticks, -1, 0, 
        #                  color=background_fill_colors[0], 
        #                  alpha=background_fill_alphas[0],
        #                  zorder=2,)
    
    ax.tick_params(axis='both', which='both', zorder=10)
    plt.savefig(filename+'.png', dpi=dpi)
    
    if show: plt.show()
    
    return fig, axs

#%%
def stacked_bar_plot(dataframe, 
                       y_ticks=[], x_ticks=[], 
                       ylim=[],
                       ax=None,
                       colors=None, 
                       hatch_patterns=('\\', '//', '|', 'x',),
                       colormap=None,
                       metric_total_values=[], metric_units=[],
                       n_minor_ticks=1,
                       y_label='', y_units='',
                       linewidth=0.8,
                       filename='stacked_bar_plot',
                       dpi=600,
                       fig_width=7,
                       fig_height=5.5*1.1777,
                       show_totals=False,
                       totals=[],
                       sig_figs_for_totals=3,
                       units_list=[],
                       totals_label_text=r"$\bfsum:$",
                       xlabelpad=5,
                       ylabelpad=5,
                       xticks_fontsize = 17,
                       ylabel_fontsize = 19,
                       yticks_fontsize = 17,
                       default_fontsize = 17,
                       label_wrapsize = 8,
                       show=False,
                       subplot_padding = 5,
                       bar_width=0.6,
                       xticks_fontcolor='black',
                       rotate_xticks=0.,
                       ):
    # axs = plt.subplot(1, 2, 2,
    #                         # gridspec_kw=gridspec_kw,
    #                         )
    
    fig = plt.gcf()
    
    plt.rcParams['font.sans-serif'] = "Arial Unicode"
    plt.rcParams['font.size'] = str(default_fontsize)
    
    ax1 = dataframe.T.plot(kind='bar', stacked=True, edgecolor='k', linewidth=linewidth,
                          color=colors,
                          colormap=colormap,
                          # facecolor="white",
                           # use_index=False,
                           ax=ax,
                          # rot=rotate_xticks,
                           width=bar_width,
                          )
    # ax1.xticks(rotation=rotate_xticks)
      # label.set_ha('right')
  
    if not ax: ax = ax1
    ax.set_facecolor("white")
    
    fig = plt.gcf()
    
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)
    # fig.tight_layout(pad=subplot_padding)
    
    
    ax.set_yticks(y_ticks,)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val}%'))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
    
    
    ax.containers[:4]
    # print(ax.containers)
    bars = [thing for thing in ax.containers if isinstance(thing, BarContainer)]
    
    
    used_facecolors = []
    used_hatches_dict = {}
    # print(len(bars))
    bar_hatch_dict = {}
    bar_num = 0
    patterns = itertools.cycle(hatch_patterns)
    for bar in bars:
        # print(len(bar))
        for patch in bar:
            
            if not bar_num in bar_hatch_dict.keys():
                curr_facecolor = patch.get_facecolor()
                
                if not curr_facecolor in used_hatches_dict.keys():
                    used_hatches_dict[curr_facecolor] = []
                    
                if curr_facecolor in used_facecolors:
                    used_hatches = used_hatches_dict[curr_facecolor]
                    curr_hatch = next(patterns)
                    i=100
                    while curr_hatch in used_hatches:
                        i+=1
                        # print(True)
                        curr_hatch = next(patterns)
                        if i>100:
                            break
                    patch.set_hatch(curr_hatch)
                    bar_hatch_dict[bar_num] = curr_hatch
                    used_hatches.append(curr_hatch)
                else:
                    bar_hatch_dict[bar_num] = None
                used_facecolors.append(curr_facecolor)
            else:
                if bar_hatch_dict[bar_num] is not None:
                    if curr_facecolor in used_facecolors:
                        patch.set_hatch(bar_hatch_dict[bar_num])
        bar_num+=1
    # print(bar_hatch_dict)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', edgecolor='white')

    ax.set_ylabel(y_label + " [" + y_units + "]", fontsize=ylabel_fontsize, labelpad=ylabelpad)
    
    ax.xaxis.labelpad=xlabelpad
    
    # ax.set_xlabel(x_labels, fontsize=14)
    
    wrap_labels(ax,label_wrapsize)
    
    ax.axhline(y=0,  color='k', linestyle='-', linewidth=linewidth)
    
    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        direction='inout',
        # right=True,
        width=1,
        labelsize=yticks_fontsize,
        )

    ax.tick_params(
        axis='y',          
        which='major',      
        length=5,
        )

    ax.tick_params(
        axis='y',          
        which='minor',      
        length=3,
        )
    
    
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        direction='inout',
        # right=True,
        width=1,
        labelsize=xticks_fontsize,
        colors=xticks_fontcolor,
        )
    
    
    ax2 = ax.twinx()
    
    
    if not y_ticks==[]:
        ax.set_yticks(y_ticks)
        ax2.set_yticks(y_ticks)
        l = ax.get_ylim()
        l2 = ax2.get_ylim()
        f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        ticks = f(ax.get_yticks())
        ax2.yaxis.set_major_locator(FixedLocator(ticks))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
    
    else:
        ax2.set_yticks(ax.get_y_ticks())
        l = ax.get_ylim()
        l2 = ax2.get_ylim()
        f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        ticks = f(ax.get_yticks())
        ax2.yaxis.set_major_locator(FixedLocator(ticks))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        
    ax2.tick_params(
        axis='y',          
        which='both',      
        direction='in',
        # right=True,
        labelright=False,
        width=1,
        )
    
    if ylim: 
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([min(y_ticks), max(y_ticks)])
    # plt.tight_layout()
    
    if show_totals:
        
        num_x_points = len(dataframe.columns)
        distance_between_x_points = 1./num_x_points
        start_x_coord = distance_between_x_points/2.
        
        ax.annotate(
                xy=(start_x_coord-0.75*distance_between_x_points, 
                1.1), 
                text=totals_label_text, 
                # fontsize=14, 
                ha='center', va='center',
                xycoords='axes fraction',
                 # transform=plt.gcf().transFigure,
                 )
        
        
        for i in range(num_x_points):
            ax.annotate(
                    xy=(start_x_coord+i*distance_between_x_points, 
                    1.1), 
                    text=get_rounded_str(totals[i], sig_figs_for_totals), 
                    # fontsize=14, 
                    ha='center', va='center',
                    xycoords='axes fraction',
                     # transform=plt.gcf().transFigure,
                     )
            ax.annotate(
                    xy=(start_x_coord+i*distance_between_x_points, 
                    1.05), 
                    text=units_list[i], 
                    # fontsize=14, 
                    ha='center', va='center',
                    xycoords='axes fraction',
                     # transform=plt.gcf().transFigure,
                     )
    
    for label in ax1.get_xticklabels():
        label.set_rotation(rotate_xticks)
      
    plt.savefig(filename+'.png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(),
                transparent=False)
    
    if show: plt.show()
    
    return fig, ax, ax2
    # patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    # bars = ax.bar([0,5], [0,5])
    # for bar, pattern in zip(bars, patterns):
        # bar.set_hatch(pattern)
    
    # plt.xlabel(list(df.columns), weight='bold')

        
#%% Miscellaneous
def Round_off(N, n): # function Round_off from https://www.geeksforgeeks.org/round-off-number-given-number-significant-digits/#
    b = N
    # c = floor(N)
    # Counting the no. of digits 
    # to the left of decimal point 
    # in the given no.
    i = 0;
    while(b >= 1):
        b = b / 10
        i = i + 1
    d = n - i
    b = N
    b = b * (10**d)
    e = b + 0.5
    if (float(e) == float(ceil(b))):
        f = (ceil(b))
        h = f - 2
        if (h % 2 != 0):
            e = e - 1
    j = floor(e)
    m = (10**d)
    j = j / m
    return j

def count_no_of_digits_in_str_num(str_num):
    return len(str_num) - str_num.count('.')

def remove_ending_0(str_num):
    if str_num[-1]=='0':
        str_num = str_num[:-1]
    return str_num

def remove_ending_decimal_point(str_num):
    if str_num[-1]=='.':
        str_num = str_num[:-1]
    return str_num

def get_exp_str_num(str_exp_num):
    exp_str_exp_num = ''
    for i in str_exp_num:
        exp_str_exp_num+=map_superscript_str_numbers[i]
    return exp_str_exp_num

def convert_OOM_notation_e_to_10_in_str_num(str_num):
    e_notations = ('e+0', 'e+', 'e-0', 'e-')
    for e_n in e_notations:
        if e_n in str_num:
            e_n_index_in_string = str_num.index(e_n)
            str_exp_num = str_num[e_n_index_in_string+len(e_n):]
            exp_str_exp_num = get_exp_str_num(str_exp_num)
            ten_n = f' \u00D710{exp_str_exp_num}'
            str_num = str_num.replace(e_n+str_exp_num, ten_n)
        else:
            pass
    return str_num

def get_rounded_str(num, sig_figs):
    # rounded_str = remove_ending_decimal_point(remove_ending_0(str(Round_off(num,sig_figs))))
    rounded_str = remove_ending_0(str(Round_off(num,sig_figs)))
    n_digits = count_no_of_digits_in_str_num(rounded_str)
    if n_digits<sig_figs:
        while not n_digits==sig_figs:
          rounded_str+='0'  
          n_digits+=1
    else:
        rounded_str = convert_OOM_notation_e_to_10_in_str_num(f'{num:.{sig_figs}g}')
    return rounded_str