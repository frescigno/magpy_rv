#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Planetary plots

Contains:
    Mass-Radius diagram
    (lum-disance diagram??)

Author: Federica Rescigno
Version: 14.06.2022
'''

from numpy import NaN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas
from pathlib import Path
import math

G_grav = 6.67398e-11
M_sun = 1.98892e30
M_jup = 1.89813e27
M_ratio = M_sun / M_jup
Mu_sun = 132712440018.9
seconds_in_day = 86400
AU_km = 1.4960 * 10 ** 8
M_jup_to_ear = 317.828133
R_jup_to_ear = 11.209


datafile = Path('/Users/frescigno/Desktop/GP code/MAGPy final push/MR_plot_data/exoplanet.eu_catalog.csv')
trackfile = Path('/Users/frescigno/Desktop/GP code/MAGPy final push/MR_plot_data/LZeng_tracks/Zeng2016_mrtable2.txt')




def DataInput(datafile, source="EU"):
    '''
    Parameters
    ----------
    data_file : string
        Path to the csv file containing the data
    source : string
        Name of the source of the data. The default is "EU".
                
    Returns
    -------
    mass_data : pandas.DataFrame
        Dataframe containing the data that has both mass and radius determinatin
    '''
        
    data = pandas.read_csv(datafile)
    #print(data)
        
    # Get data only for planets with mass and radius fields
    if source.startswith("eu") or source.startswith("EU"):
        data['mass'].update(data.pop('mass_sini'))
        data['mass_error_min'].update(data.pop('mass_sini_error_min'))
        data['mass_error_max'].update(data.pop('mass_sini_error_max'))
        #print(data)
        mass_data = data[~data['mass'].isnull() & ~data['radius'].isnull()]
        
        # change all nan value in mass and radius error min and max to zero
        mass_data['radius_error_min'] = mass_data['radius_error_min'].replace(NaN, 0)
        mass_data['radius_error_max'] = mass_data['radius_error_max'].replace(NaN, 0)
        mass_data['mass_error_min'] = mass_data['mass_error_min'].replace(NaN, 0)
        mass_data['mass_error_max'] = mass_data['mass_error_max'].replace(NaN, 0)
        #print(mass_data)
        
    return(mass_data)



def selection(data, mass_limits=[0.,100.], rad_limits=[0.,100.], earth_units=True):
    ''' 
    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the data that has both mass and radius determinatin
    mass_limits : list of floats
        Limits of the mass range. The default is [0.,10.].
    rad_limits : list of floats
        Limits of the radius range. The default is [0.,10].
    earth_units : True or False
        If True, the data is in earth units. The default is True.
    
    Returns
    -------
    selected_data : pandas.DataFrame
        Dataframe containing the data that has both mass and radius determinations within expected limits
    '''
    
    assert mass_limits[0] < mass_limits[1], "mass_limits[0] must be smaller than mass_limits[1]"
    assert rad_limits[0] < rad_limits[1], "rad_limits[0] must be smaller than rad_limits[1]"
    
    if earth_units:
        data['mass'] = data['mass'] * M_jup_to_ear
        data['mass_error_min'] = data['mass_error_min'] * M_jup_to_ear
        data['mass_error_max'] = data['mass_error_max'] * M_jup_to_ear
        data['radius'] = data['radius'] * R_jup_to_ear
        data['radius_error_min'] = data['radius_error_min'] * R_jup_to_ear
        data['radius_error_max'] = data['radius_error_max'] * R_jup_to_ear
    
    # start with selecting the proper parameter space
    selected_data = data[(data['mass'] > mass_limits[0]) & (data['mass'] < mass_limits[1])
                         & (data['radius'] > rad_limits[0]) & (data['radius'] < rad_limits[1])
                         & (data['mass_error_min'] > 0.) & (data['mass_error_max'] > 0.)
                         & (data['radius_error_min'] > 0.) & (data['radius_error_max'] > 0.)
                         & (data['mass_error_min'] < 20.) & (data['mass_error_max'] < 20.)]
    
    # then eliminate the stars with no characterisation or theroratical detections
    selected_data = selected_data[~selected_data['detection_type'].isnull()
                                  & ~selected_data['mass_detection_type'].isnull()
                                  & ~selected_data['radius_detection_type'].isnull()]
    selected_data = selected_data[~selected_data['star_mass'].isnull() & ~selected_data['star_radius'].isnull()
                                  & ~selected_data['star_teff'].isnull()]
    
    return selected_data
    
        


class MR_plot:
    
    def __init__(self):
        '''
        Initialize the class
        '''
        self.xticks = [0.5, 1, 2, 5, 10, 20, 40, 80, 100]
        self.yticks = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4,]
        self.yticks = [1, 2, 3, 4, 5, 6, 7, 8]
        self.xlims = [0.1, 100.]
        self.ylims = [0.5, 9.]
        self.xy_labels = [19.0, 4.2]
        self.xlabel = "Mass in Earth Masses"
        self.ylabel = "Radius in Earth Radii"
        
        self.colorbar_xvector = [1, 3, 10, 30, 100, 300, 1000, 3000, 6000]
        self.add_overplot = 0.0
        

        self.font_label = 18
        self.font_planet_name = 10
        self.font_tracks =12
        self.font_my_planet = 18
        self.font_USP_name = 12
        self.font_Solar_name =15
        self.font_my_planet = 18

        
        
        self.markersize = 6
        self.markersize = 8
        self.z_offset = 0.0
        self.marker_point = "o"
        
        self.logM = True
        self.logR = False
        self.add_lzeng_tracks = True
        
        self.add_jupiter_densities = False
        self.jupiter_densities_list = [1.0, 0.50, 0.25, 0.10, 0.05, 0.030]
        
        self.plot_size = [8,6]

        self.default_plot_parameters = {'x_pos':None, 'y_pos': None, 'rotation':None,
                                        'use_box':False, 'cmap':'gist_heat', 'color':0.00,
                                        'alpha':0.8, 'linestyle':'-', 'label':"100% Fe"}
        
        self.tracks_on_top = True
        
        self.lzeng_tracks = np.genfromtxt(trackfile, skip_header=1, \
         names = ['Mearth','100_fe','75_fe','50_fe','30_fe','25_fe','20_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip'] )
        
        self.lzeng_plot_list = ['100_fe','75_fe','50_fe','25_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip']
        self.lzeng_plot_parameters = {
            '100_fe':         {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.00, 'alpha':0.8, 'linestyle':'-', 'label':'100% Fe'},
            '75_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.25, 'alpha':0.8, 'linestyle':'-', 'label':'75% Fe'},
            '50_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.50, 'alpha':0.8, 'linestyle':'-', 'label':'50% Fe'},
            '30_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.70, 'alpha':0.8, 'linestyle':'-', 'label':'30% Fe'},
            '25_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.75, 'alpha':0.8, 'linestyle':'-', 'label':'25% Fe'},
            '20_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.80, 'alpha':0.8, 'linestyle':'-', 'label':'20% Fe'},
            'rocky':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'Greens',    'color':0.50, 'alpha':0.8, 'linestyle':'-', 'label':'100% Rocky'},
            #'75_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.25, 'alpha':0.8, 'linestyle':'-', 'label':'75% Fe, 25% MgSiO$_{3}$'},
            #'50_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.50, 'alpha':0.8, 'linestyle':'-', 'label':'50% Fe, 50% MgSiO$_{3}$'},
            #'30_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.70, 'alpha':0.8, 'linestyle':'-', 'label':'30% Fe, 70% MgSiO$_{3}$'},
            #'25_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.75, 'alpha':0.8, 'linestyle':'-', 'label':'25% Fe, 75% MgSiO$_{3}$'},
            #'20_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.80, 'alpha':0.8, 'linestyle':'-', 'label':'20% Fe, 80% MgSiO$_{3}$'},
            #'rocky':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'Greens',    'color':0.50, 'alpha':0.8, 'linestyle':'-', 'label':'100% MgSiO$_{3}$'},
            '25_h2o':         {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':True , 'cmap':'winter', 'color':0.75, 'alpha':0.8, 'linestyle':'-', 'label':'25% H$_{2}$O'},
            '50_h2o':         {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':True , 'cmap':'winter', 'color':0.50, 'alpha':0.8, 'linestyle':'-', 'label':'50% H$_{2}$O'},
            '100_h2o':        {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':True , 'cmap':'winter', 'color':0.00, 'alpha':0.8, 'linestyle':'-', 'label':'100% H$_{2}$O'},
            'cold_h2_he':     {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'winter', 'color':0.90, 'alpha':0.8, 'linestyle':'-', 'label':'Cold H$_{2}$/He'},
            'max_coll_strip': {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'binary', 'color':1.00, 'alpha':0.2, 'linestyle':'-', 'label':'', 'fill_below':True}
        }
        
        
        self.radius_gap_on_top = False
        self.add_radius_gap = True
        self.radius_gap_parameters = {'x_pos':None, 'y_pos': 1.95, 'rotation':None, 'cmap':'Blues', 'color':0.70, 'alpha':0.2, 'label':'Radius gap'}
        self.radius_gap = [1.7, 0.2]
        self.radius_gap_shaded = True
        self.radius_label_position = ['bottom', 'left']


        self.colorbar_axes_list=[0.10, 0.65, 0.03, 0.30]
        self.colorbar_ticks_position = 'right'
        
        csfont = {'fontname':'Times New Roman'}
        matplotlib.rc('font',**{'family':'serif','serif':['Times New Roman'], 'size':18})
    
    def get_radius_gap(self):
        self.radius_gap_x = np.arange(self.xlims[0]-self.xlims[0]/2., self.xlims[1]+0.1, 0.01)
        self.radius_gap_y = np.ones(len(self.radius_gap_x))*self.radius_gap[0]
    
    def setup(self):
        
        self.xrange = self.xlims[1]-self.xlims[0]
        self.yrange = self.ylims[1]-self.ylims[0]
        self.fig, self.ax1 = plt.subplots(1, figsize=(self.plot_size[0], self.plot_size[1]))
        #self.fig.tight_layout()
        self.ax1.set_xlim(self.xlims)
        self.ax1.set_ylim(self.ylims)
        self.ax1.set_xlabel('Mass [Earth Mass]')
        self.ax1.set_ylabel('Radius [Earth Radius]')
        
        if self.logM:
            self.ax1.set_xscale('log')
            self.ax1.set_xticks(self.xticks)
            self.ax1.minorticks_off()
            self.ax1.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            self.ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
            self.ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(x)))
        if self.logR:
            self.ax1.set_yscale('log')
            self.ax1.set_yticks(self.yticks)
            self.ax1.minorticks_off()
            self.ax1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            self.ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
            self.ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(x)))

        self.get_radius_gap()
    
    def interpolate_line_value(self, mass, radius, x_pos=None, y_pos=None,
        default_position = 'top_right'):

        if default_position == 'top_right':
            ind = np.where((radius<=self.ylims[1]) & (mass<=self.xlims[1]))[0][-1]
        if default_position == 'bottom_left':
            ind = np.where((radius>=self.ylims[0]) & (mass>=self.xlims[0]))[0][0]

        x1 = mass[ind]
        x2 = mass[ind+1]
        y1 = radius[ind]
        y2 = radius[ind+1]

        if y1 == y2:
            if x_pos is None:
                return self.xy_labels[0], y1
            else:
                return x_pos, y1

        if x1 == x2:
            if y_pos is None:
                return x1, self.xy_labels[1]
            else:
                return x1, y_pos

        if x_pos:
            y_out = y1 + (x_pos-x1)*(y2-y1)/(x2-x1)
            return x_pos, y_out

        if y_pos:
            x_out = x1 + (y_pos-y1)*(x2-x1)/(y2-y1)
            return x_out, y_pos

        if x2>self.xlims[1]:
            # Case: the track is ending on the right side of the plot
            x_out = self.xy_labels[0]
            y_out = y1 + (x_out-x1)*(y2-y1)/(x2-x1)
        else:
            # Case: the track is ending on the upper part of the plot
            y_out = self.xy_labels[1]
            x_out = x1 + (y_out-y1)*(x2-x1)/(y2-y1)

        return x_out, y_out
    
    def text_slope_match_line(self, ax, xdata, ydata, pos, nearly_vertical=False):

        # find the slope

        if nearly_vertical:
            ind = np.where(ydata > pos)[0][0]
        else:
            ind = np.argmin(np.abs(xdata-pos))

        x1 = xdata[ind-1]
        x2 = xdata[ind+1]
        y1 = ydata[ind-1]
        y2 = ydata[ind+1]

        p1 = np.array((x1, y1))
        p2 = np.array((x2, y2))

        # get the line's data transform
        #ax = ax.get_axes()

        sp1 = ax.transData.transform_point(p1)
        sp2 = ax.transData.transform_point(p2)

        rise = (sp2[1] - sp1[1])
        run = (sp2[0] - sp1[0])

        return math.degrees(math.atan(rise/run))
    
    def add_tracks(self, mass, radius, key_val=None):
        if key_val is None:
            key_val = self.default_plot_parameters
        
        try:
            if not (key_val['x_pos'] or key_val['y_pos']):
                x_pos, y_pos = self.interpolate_line_value(mass, radius)
            elif key_val['x_pos'] and key_val['y_pos']:
                x_pos = key_val['x_pos']
                y_pos = key_val['y_pos']
            elif key_val['x_pos']:
                x_pos, y_pos = self.interpolate_line_value(mass, radius, x_pos=key_val['x_pos'])
            elif key_val['y_pos']:
                x_pos, y_pos = self.interpolate_line_value(mass, radius, y_pos=key_val['y_pos'])
            if key_val['rotation']:
                rotation = key_val['rotation']
            else:
                rotation = self.text_slope_match_line(self.ax1, mass, radius, x_pos)
        except:
            print(key_val['label'] + " composition track outside the boundaries of the plot")
            return

        bbox_props = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='w', pad=0.0)

        color_map = plt.get_cmap(key_val['cmap'])
        color = color_map(key_val['color'], alpha=key_val['alpha'])
        color_noalpha = color_map(key_val['color'], alpha=1.0)
        
        if 'overplot' in key_val:
            line = self.ax1.plot(mass, radius, color=color, zorder=950, ls=key_val['linestyle'])
        else:
            line = self.ax1.plot(mass, radius, color=color, zorder=0, ls=key_val['linestyle'])
            
        if key_val['use_box']:
            self.ax1.annotate(key_val['label'], xy=(x_pos, y_pos),
                              xytext=(0, -6), textcoords='offset points', ha='right', va='top', 
                              color=color_noalpha, zorder=1000, rotation=rotation, rotation_mode="anchor",
                              fontsize=self.font_tracks,  fontweight='bold', bbox=bbox_props, fontstretch='extra-condensed')
        else:
            self.ax1.annotate(key_val['label'], xy=(x_pos, y_pos),
                              xytext=(0, -6), textcoords='offset points', ha='right', va='top',
                              color=color_noalpha, zorder=1000, rotation=rotation, rotation_mode="anchor",
                              fontsize=self.font_tracks, fontweight='bold', fontstretch='extra-condensed' )#, bbox=bbox_props)
            if 'fill_below' in key_val:
                self.ax1.fill_between(mass, 0, radius, color=color, alpha=0.15)
            if 'fill_between' in key_val:
                self.ax1.fill_between(mass, radius-0.01,radius+0.01, color=color, zorder=0, alpha=0.5)
        
    
    def plot_lzeng_tracks(self):
        for key_name in self.lzeng_plot_list:
            self.add_tracks(self.lzeng_tracks['Mearth'],
                    self.lzeng_tracks[key_name],
                    self.lzeng_plot_parameters[key_name])
    
    def plot_radius_gap(self):
        if  self.radius_gap_on_top:
            radius_z_order = self.z_offset + 1000.0
        else:
            radius_z_order = self.z_offset
        xytext = [0, 0]
        key_val = self.radius_gap_parameters
        
        if 'top' in self.radius_label_position:
            va='top'
            xytext[1] = 3
            y_axis = self.radius_gap_y+self.radius_gap[1]

        if 'bottom' in self.radius_label_position:
            va='bottom'
            xytext[1] = -6
            y_axis = self.radius_gap_y-self.radius_gap[1]

        if 'left' in self.radius_label_position:
            ha='left'
            xytext[0] = 3
            x_pos = self.xlims[0]

        if 'right' in self.radius_label_position:
            ha='right'
            xytext[0] = -3
            x_pos = self.xlims[1]

        if key_val['x_pos']:
            x_pos = key_val['x_pos']

        if key_val['y_pos']:
            y_pos = key_val['y_pos']
        else:
            x_pos, y_pos = self.interpolate_line_value(self.radius_gap_x, y_axis, x_pos=x_pos)

        if key_val['rotation']:
            rotation = key_val['rotation']
        else:
            rotation = self.text_slope_match_line(self.ax1, self.radius_gap_x, y_axis, x_pos)
        
        color_map = plt.get_cmap(key_val['cmap'])
        color = color_map(key_val['color'], alpha=key_val['alpha'])
        color_noalpha = color_map(key_val['color'], alpha=1.0)
        self.ax1.fill_between(self.radius_gap_x, self.radius_gap_y-self.radius_gap[1],
                              self.radius_gap_y+self.radius_gap[1], color=color, zorder=0)
        
        self.ax1.annotate(key_val['label'], xy=(x_pos, y_pos), \
                         xytext=(xytext[0], xytext[1]), textcoords='offset points', ha=ha, va=va, \
                         color=color_noalpha, zorder=1000+radius_z_order, rotation=rotation, rotation_mode="anchor",
                     fontsize=self.font_tracks, weight='bold')
    
    def add_data(self, data):
        
        n_planets = len(data)
        for ind in range(0,n_planets):
            xpos = data['mass'].iloc[ind]
            ypos = data['radius'].iloc[ind]
            m_err_max = data['mass_error_max'].iloc[ind]
            m_err_min = data['mass_error_min'].iloc[ind]
            r_err_max = data['radius_error_max'].iloc[ind]
            r_err_min = data['radius_error_min'].iloc[ind]
            
            pl_name = data['# name'].iloc[ind]
            m_det_type = data['mass_detection_type'].iloc[ind]
            pl_P = data['orbital_period'].iloc[ind]
            
            marker_point = self.marker_point
            self.ax1.errorbar(xpos, ypos, xerr=([m_err_min], [m_err_max]),
                              marker=marker_point, mfc='white', markersize=self.markersize, color='grey', alpha=0.5)
            self.ax1.errorbar(xpos, ypos, yerr=([r_err_min], [r_err_max]),
                              marker=marker_point, mfc='white', markersize=self.markersize, color='grey', alpha=0.5)
            self.ax1.plot(xpos, ypos,
                          marker=marker_point, markersize=self.markersize, color='grey', alpha=0.5)
    
    def add_my_pl(self, mass, mass_error_min, mass_error_max, radius, radius_error_min, radius_error_max, name):
        
        n_planets = len(mass)
        for ind in range(0,n_planets):
            xpos = mass[ind]
            ypos = radius[ind]
            m_err_min = mass_error_min[ind]
            m_err_max = mass_error_max[ind]
            r_err_min = radius_error_min[ind]
            r_err_max = radius_error_max[ind]
            pl_name = name[ind]
            
            marker_point = '*'
            
            self.ax1.errorbar(xpos, ypos, xerr=([m_err_min], [m_err_max]), color='purple',
                              zorder=10e3, marker=marker_point, mfc='white', markersize=self.markersize, lw=2)
            self.ax1.errorbar(xpos, ypos, yerr=([r_err_min], [r_err_max]), color='purple',
                              zorder=10e3, marker=marker_point, mfc='white', markersize=self.markersize, lw=2)
            self.ax1.plot(xpos, ypos, color='purple', zorder=10e3+0.3, marker=marker_point,
                          mfc='purple', mec='k', mew = 2 , markersize=self.markersize*2, lw=4)
            if xpos*0.98 < self.xlims[0] or xpos*1.02 > self.xlims[1] or ypos*0.98 < self.ylims[0] or ypos > self.ylims[1]: continue
            bbox_props = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='k', pad=0.2)
            xytext = [-7, 10]
            self.ax1.annotate(pl_name, xy=(xpos, ypos),
                xytext=xytext, textcoords='offset points', ha='right', va='bottom',
                color='k', fontsize=self.font_my_planet, zorder=10e3-10,
                annotation_clip=True, bbox=bbox_props)
        
            
    
    def add_solar_system(self):
        earth = "Earth"
        venus = "Venus"
        bbox_props = dict(boxstyle="square", fc="w", alpha=0.9, edgecolor='b', pad=0.1)
        
        self.ax1.plot([0.815, 1.00],[0.949,1.00],'ob', markersize=self.markersize+4, marker='*', zorder= 10000+ self.z_offset)
        self.ax1.annotate(earth, xy=(1.0, 1.0),
                     xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                     color='b', fontsize=self.font_Solar_name, zorder= 10000+ self.z_offset,
                     annotation_clip=True, bbox=bbox_props, stretch='condensed')
        self.ax1.annotate(venus, xy=(0.815, 0.949),
                     xytext=(5, 5), textcoords='offset points', ha='right', va='bottom',
                     color='b', fontsize=self.font_Solar_name, zorder= 10000+ self.z_offset,
                     annotation_clip=True, bbox=bbox_props, stretch='condensed')
        
        
    
    def make_plot(self, dataset, mass, mass_error_min, mass_error_max, radius, radius_error_min, radius_error_max, name):
        
        self.setup()
        if self.add_lzeng_tracks:
            self.plot_lzeng_tracks()
        if self.add_radius_gap:
            self.plot_radius_gap()
        
        self.add_data(dataset)
        self.add_solar_system()
        
        self.add_my_pl(mass, mass_error_min, mass_error_max, radius, radius_error_min, radius_error_max, name)

        plt.savefig('MRplot2.pdf')
        plt.show()

dataset_nosel = DataInput(datafile)
datasets = selection(dataset_nosel)
print()
print()
print()
print(len(datasets))
#print("###################\n\n\n",datasets['mass'][0])
mass = [6.8, 58.4]
massu = [2, 2]
massl =[2, 2]
radius = [3.02, 7.8]
radiusu =[0.2, 0.2]
radiusl=[0.2, 0.2]
name=['TOI-2134b', 'TOI-2134c']
MRplot = MR_plot()
MRplot.make_plot(datasets, mass, massl, massu, radius, radiusl, radiusu, name)