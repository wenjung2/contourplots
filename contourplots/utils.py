# -*- coding: utf-8 -*-
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
from matplotlib.ticker import AutoMinorLocator, LinearLocator, FixedLocator
from matplotlib.ticker import FuncFormatter

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
                                  figwidth=4.5,
                                  dpi=600,
                                  cmap='viridis',
                                  z_marker_color='b',
                                  z_marker_type='v',
                                  axis_title_fonts={'size': {'x': 12, 'y':12, 'z':12, 'w':12},},
                                  gap_between_figures=20., 
                                  fps=3, # animation frames (z values traversed) per second
                                  n_loops='inf', # the number of times the animated contourplot should loop animation over z; infinite by default
                                  animated_contourplot_filename='animated_contourplot',
                                  keep_frames=False, # leaves frame PNG files undeleted after running; False by default
                                 ):
    results = np.array(w_data_vs_x_y_at_multiple_z)
    
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
        im = ax.contourf(x_data, y_data, results[z_index],
                          cmap=cmap,
                         levels=w_levels,
                         )
  
        clines = ax.contour(x_data, y_data, results[z_index],
                   levels=w_ticks,
                    colors='black',
                   linewidths=w_tick_width)
        
        ax.clabel(clines, 
                   w_ticks,
                   fmt=fmt_clabel, 
                  fontsize=10,
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
        
        cbar = plt.colorbar(im, cax=cax, 
                     ticks = w_levels).set_label(label=w_label + " [" + w_units + "]", 
                                                           size=axis_title_fonts['size']['w'],
                                                           loc='center',
                                                           **fontname
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
def box_and_whiskers_plot(uncertainty_data, 
                          baseline_value=None, 
                          range_for_comparison=(None, None),
                          values_for_comparison=[],
                          n_minor_ticks=1,
                          y_label='Metric',
                          y_units='units',
                          y_ticks=[],
                          boxcolor="#A97802",
                          save_file=True,
                          filename='box_and_whiskers_plot',
                          dpi=600,
                          fig_width=1.5,
                          fig_height=4.,
                          box_width=1.5,
                          ):
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = "14"

    gridspec_kw={'height_ratios': [1, 20]},

    fig, axs = plt.subplots(1, 1, constrained_layout=True, 
                            # gridspec_kw=gridspec_kw,
                            )
    ax = axs
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

    ax.plot(1, baseline_value, c='k', 
            # label='.',
            marker='D', 
                markersize=6, 
                markerfacecolor='w',
                  markeredgewidth=0.8,
                 zorder=100)

    if not (range_for_comparison==() or range_for_comparison==[] or range_for_comparison==None):
        ax.fill_between([0, 1, 2], [range_for_comparison[0]], [range_for_comparison[1]], color="#c0c1c2", alpha=1., zorder=1)

    if not (values_for_comparison==() or values_for_comparison==[] or values_for_comparison==None):
        1
    ax.yaxis.set_minor_locator(AutoMinorLocator(n_minor_ticks+1))


    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

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
                  {'fontname':'Calibri'}, fontsize=14, 
               # fontweight='bold',
               )




    plt.savefig(filename+'.png', dpi=dpi)

#%%
def stacked_bar_plot(dataframe, 
                       y_ticks=[], x_ticks=[], 
                       colors=None, 
                       colormap=None,
                       metric_total_values=[], metric_units=[],
                       n_minor_ticks=1,
                       y_label='', y_units='',
                       linewidth=0.8,
                       filename='stacked_bar_plot',
                       dpi=600,
                       fig_width=7,
                       fig_height=5.5):
    
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
    
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', edgecolor='white')

    ax.set_ylabel(y_label + " [" + y_units + "]", fontsize=14)
    
    
    # ax.set_xlabel(x_labels, fontsize=14)
    import textwrap
    def wrap_labels(ax, width, break_long_words=False):
        labels = []
        for label in ax.get_xticklabels():
            text = label.get_text()
            labels.append(textwrap.fill(text, width=width,
                          break_long_words=break_long_words))
        ax.set_xticklabels(labels, rotation=0, fontsize=14, weight='bold')
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

        
    