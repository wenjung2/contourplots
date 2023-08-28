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

defaults_dict ={'colors':
                {'Guest_Group_TEA_Breakdown': ['#7BBD84', '#F7C652', '#63C6CE', '#94948C', '#734A8C', '#D1C0E1', '#648496', '#B97A57', '#D1C0E1', '#F8858A', '#F8858A', ]}}

def wrap_labels(ax, width, break_long_words=False, fontsize=14):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0, fontsize=fontsize, weight='bold')

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
                                  fontname={'fontname':'Arial'},
                                  figwidth=3.9,
                                  dpi=600,
                                  cmap='viridis',
                                  extend_cmap='neither',
                                  cmap_over_color=None,
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
                                  n_minor_ticks = 1,
                                  cbar_n_minor_ticks = 4,
                                  comparison_range=[],
                                  comparison_range_hatch_pattern='///',
                                    ):
    results = np.array(w_data_vs_x_y_at_multiple_z)
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = "14"
    def create_frame(z_index):
        fig, axs = plt.subplots(2, 1, constrained_layout=True, 
                                gridspec_kw=gridspec_kw)
        fig.set_figwidth(figwidth)
        ax = axs[0]
        a = [z_data[z_index]]
        ax.hlines(1,1,1)
        ax.set_xlim(min(z_ticks), max(z_ticks))
        ax.set_ylim(0.5,1.5)
        
        
        
        ax.xaxis.tick_top()
        ax.set_xlabel(z_label + " [" + z_units + "]",  
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
        
        ax = axs[1]
        if cmap_over_color is not None:
            cmap.set_over(cmap_over_color)
        
            
        im = ax.contourf(x_data, y_data, results[z_index],
                          cmap=cmap,
                         levels=w_levels,
                         extend=extend_cmap
                         
                         )
        
        
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))
        # ########--########
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            direction='inout',
            # right=True,
            width=0.65,
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
            )
        ax.tick_params(
            axis='x',          
            which='major',      
            length=7,
            )

        ax.tick_params(
            axis='x',          
            which='minor',      
            length=3.5,
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
        
        if not comparison_range==[]:
            [m1,n1] = np.where((results[z_index] > comparison_range[0]) & (results[z_index] < comparison_range[1]))
            # [m2,n2] = np.where(results[z_index] < 7.5)
            
            z1 = np.zeros(results[z_index].shape)
            z1[m1,n1] = 99
            
            # print(z1,)
            plt.rcParams['hatch.linewidth'] = 0.6
            plt.rcParams['hatch.color'] = 'white'
            cs = ax.contourf(x_data, y_data, z1 ,1 , hatches=['', comparison_range_hatch_pattern],  alpha=0.,
                             # extend=extend_cmap,
                             )
            
        clines = ax.contour(x_data, y_data, results[z_index],
                   levels=w_ticks,
                    colors='black',
                   linewidths=w_tick_width)
        
        ax.clabel(clines, 
                   w_ticks,
                   fmt=fmt_clabel, 
                  fontsize=clabel_fontsize,
                  colors='black',
                  )
        
        if not comparison_range==[]:
            clines2 = ax.contour(x_data, y_data, results[z_index],
                       levels=comparison_range,
                        colors='white',
                       linewidths=w_tick_width)
            
            ax.clabel(clines2, 
                       comparison_range,
                       fmt=fmt_clabel, 
                      fontsize=clabel_fontsize,
                      colors='black',
                      )
        
        ax.set_ylabel(y_label + " [" + y_units + "]",  
                      fontsize=axis_title_fonts['size']['y'],
                      **fontname)
        ax.set_xlabel(x_label + " [" + x_units + "]", 
                      fontsize=axis_title_fonts['size']['x'],
                      **fontname)
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        norm = mpl.colors.BoundaryNorm(w_levels, cmap.N, extend='max')
        
        # if not cbar_ticks:
        #     cbar_ticks = w_levels
        cbar = plt.colorbar(
                            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                            # im,
                            cax=cax, 
                     ticks = cbar_ticks)
        
        cbar.set_label(label=w_label + " [" + w_units + "]", 
                                              size=axis_title_fonts['size']['w'],
                                              loc='center',
                                              **fontname
                                              )
        
        # cbar.ax.set_minor_locator(AutoMinorLocator(cbar_n_minor_ticks+1))
        cbar.ax.minorticks_on()
        
        cbar.ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            direction='inout',
            # right=True,
            width=0.65,
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
        
        ax.set_title(' ', fontsize=gap_between_figures)

        plt.savefig(f'./{animated_contourplot_filename}_frame_{z_index}.png', 
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
        image = imageio.v2.imread(f'./{animated_contourplot_filename}_frame_{z_index}.png')
        frames.append(image)
    
    
    if n_loops==('inf' or 'infinite' or 'infinity' or np.inf):
        imageio.mimsave('./' + animated_contourplot_filename + '.gif',
                        frames,
                        fps=fps,
                        ) 
    else:
        imageio.mimsave('./' + animated_contourplot_filename + '.gif',
                        frames,
                        fps=fps,
                        loop=n_loops,
                        ) 
    
    if not keep_frames:
        for z_index in range(len(z_data)):
            os.remove(f'./{animated_contourplot_filename}_frame_{z_index}.png')


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
                          x_tick_fontsize=14,
                          x_tick_wrap_width=6,
                          boxcolor="#A97802",
                          save_file=True,
                          filename='box_and_whiskers_plot',
                          dpi=600,
                          fig_width=1.5,
                          fig_height=5.5,
                          box_width=1.5,
                          height_ratios = [1, 20],
                          
                          ):
    n_boxes = 1. if not hasattr(uncertainty_data[0], '__iter__') else len(uncertainty_data)
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = "14"

    gridspec_kw={'height_ratios': height_ratios,},

    fig, axs = plt.subplots(1, 1, constrained_layout=True, 
                            # gridspec_kw=gridspec_kw,
                            )
    ax = axs
    
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
        1
    
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
            labelbottom=False if x_tick_labels is None else True)
    
    if show_x_ticks and (x_tick_labels is not None):
        ax.set_xticks(ticks=list(range(1,n_boxes+1)), labels=x_tick_labels,)
        wrap_labels(ax, width=x_tick_wrap_width, fontsize=x_tick_fontsize)
    

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
        )
    
    ax.set_ylabel(y_label + " [" + y_units + "]", 
                  {'fontname':'Arial'}, fontsize=14, 
               # fontweight='bold',
               )




    plt.savefig(filename+'.png', dpi=dpi)

#%%
def stacked_bar_plot(dataframe, 
                       y_ticks=[], x_ticks=[], 
                       colors=None, 
                       hatch_patterns=('\\', '//', 'x',  '|',),
                       colormap=None,
                       metric_total_values=[], metric_units=[],
                       n_minor_ticks=1,
                       y_label='', y_units='',
                       linewidth=0.8,
                       filename='stacked_bar_plot',
                       dpi=600,
                       fig_width=7,
                       fig_height=5.5*1.1777):
    
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = "14"
    
    ax = dataframe.T.plot(kind='bar', stacked=True, edgecolor='k', linewidth=linewidth,
                          color=colors,
                          colormap=colormap,
                          # facecolor="white",
                           # use_index=False,
                          rot=0,
                          )
    
    
    ax.set_facecolor("white")
    
    fig = plt.gcf()
    
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)
    
    
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
    
    
    # plt.tight_layout()
    
    

    plt.savefig(filename+'.png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(),
                transparent=False)
    
    # patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    # bars = ax.bar([0,5], [0,5])
    # for bar, pattern in zip(bars, patterns):
        # bar.set_hatch(pattern)
    
    # plt.xlabel(list(df.columns), weight='bold')

        
    