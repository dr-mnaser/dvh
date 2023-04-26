# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 00:23:49 2023

@author: MANaser
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# dvhcalc.py
#Calculate dose volume histogram (DVH) from DICOM RT Structure/Dose data."""
# Copyright (c) 2016 gluce
# Copyright (c) 2011-2016 Aditya Panchal
# Copyright (c) 2010 Roy Keyes
# This file is part of dicompyler-core, released under a BSD license.
#    See the file license.txt included with this distribution, also
#    available at https://github.com/dicompyler/dicompyler-core/

from __future__ import division
from dicompylercore import dicomparser, dvh, dvhcalc
import numpy as np
import numpy.ma as ma
import matplotlib.path
from six import iteritems
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import os
import streamlit as st
import tempfile
import pandas as pd
import psutil

logger = logging.getLogger('dicompylercore.dvhcalc')

def get_system_usage():
    cpu_percent = psutil.cpu_percent()
    mem_percent = psutil.virtual_memory().percent
    return cpu_percent, mem_percent

# Define a function to check if the app can serve a new user based on the current resource usage
def can_serve_user():
    cpu_percent, mem_percent = get_system_usage()
    # Check if the current CPU and memory usage are below the threshold
    if cpu_percent < 80 and mem_percent < 80:
        return True
    else:
        return False

def get_dvh(structure, dose, roi, limit=None, callback=None):
    """Calculate a cumulative DVH in Gy from a DICOM RT Structure Set & Dose.
    Parameters
    ----------
    structure : pydicom Dataset
        DICOM RT Structure Set used to determine the structure data.
    dose : pydicom Dataset
        DICOM RT Dose used to determine the dose grid.
    roi : int
        The ROI number used to uniquely identify the structure in the structure
        set.
    limit : int, optional
        Dose limit in cGy as a maximum bin for the histogram.
    callback : function, optional
        A function that will be called at every iteration of the calculation.
    """
    from dicompylercore import dicomparser
    rtss = dicomparser.DicomParser(structure)
    rtdose = dicomparser.DicomParser(dose)
    structures = rtss.GetStructures()
    s = structures[roi]
    s['planes'] = rtss.GetStructureCoordinates(roi)
    s['thickness'] = rtss.CalculatePlaneThickness(s['planes'])
    hist = calculate_dvh(s, rtdose, limit, callback)
    return dvh.DVH(counts=hist,
                   bins=(np.arange(0, 2) if (hist.size == 1) else
                         np.arange(0, hist.size + 1) / 100),
                   dvh_type='differential',
                   dose_units='gy',
                   name=s['name']
                   ).cumulative


def calculate_dvh(structure, dose, limit=None, callback=None):
    """Calculate the differential DVH for the given structure and dose grid.
    Parameters
    ----------
    structure : dict
        A structure (ROI) from an RT Structure Set parsed using DicomParser
    dose : DicomParser
        A DicomParser instance of an RT Dose
    limit : int, optional
        Dose limit in cGy as a maximum bin for the histogram.
    callback : function, optional
        A function that will be called at every iteration of the calculation.
    """
    planes = structure['planes']
    logger.debug(
        "Calculating DVH of %s %s", structure['id'], structure['name'])

    # Create an empty array of bins to store the histogram in cGy
    # only if the structure has contour data or the dose grid exists
    if ((len(planes)) and ("PixelData" in dose.ds)):

        # Get the dose and image data information
        dd = dose.GetDoseData()
        id = dose.GetImageData()

        # Generate a 2d mesh grid to create a polygon mask in dose coordinates
        # Code taken from Stack Overflow Answer from Joe Kington:
        # https://stackoverflow.com/q/3654289/74123
        # Create vertex coordinates for each grid cell
        x, y = np.meshgrid(np.array(dd['lut'][0]), np.array(dd['lut'][1]))
        x, y = x.flatten(), y.flatten()
        dosegridpoints = np.vstack((x, y)).T

        maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)
        # Remove values above the limit (cGy) if specified
        if isinstance(limit, int):
            if (limit < maxdose):
                maxdose = limit
        hist = np.zeros(maxdose)
    else:
        return np.array([0])

    n = 0
    planedata = {}
    # Iterate over each plane in the structure
    for z, plane in iteritems(planes):
        # Get the dose plane for the current structure plane
        doseplane = dose.GetDoseGrid(z)
        planedata[z] = calculate_plane_histogram(
            plane, doseplane, dosegridpoints,
            maxdose, dd, id, structure, hist)
        n += 1
        if callback:
            callback(n, len(planes))
    # Volume units are given in cm^3
    volume = sum([p[1] for p in planedata.values()]) / 1000
    # Rescale the histogram to reflect the total volume
    hist = sum([p[0] for p in planedata.values()])
    hist = hist * volume / sum(hist)
    # Remove the bins above the max dose for the structure
    hist = np.trim_zeros(hist, trim='b')

    return hist


def calculate_plane_histogram(plane, doseplane, dosegridpoints,
                              maxdose, dd, id, structure, hist):
    """Calculate the DVH for the given plane in the structure."""
    contours = [[x[0:2] for x in c['data']] for c in plane]

    # If there is no dose for the current plane, go to the next plane
    if not len(doseplane):
        return (np.arange(0, maxdose), 0)

    # Create a zero valued bool grid
    grid = np.zeros((dd['rows'], dd['columns']), dtype=np.uint8)

    # Calculate the histogram for each contour in the plane
    # and boolean xor to remove holes
    for i, contour in enumerate(contours):
        m = get_contour_mask(dd, id, dosegridpoints, contour)
        grid = np.logical_xor(m.astype(np.uint8), grid).astype(np.bool)

    hist, vol = calculate_contour_dvh(
        grid, doseplane, maxdose, dd, id, structure)
    return (hist, vol)


def get_contour_mask(dd, id, dosegridpoints, contour):
    """Get the mask for the contour with respect to the dose plane."""
    doselut = dd['lut']

    c = matplotlib.path.Path(list(contour))
    grid = c.contains_points(dosegridpoints)
    grid = grid.reshape((len(doselut[1]), len(doselut[0])))

    return grid


def calculate_contour_dvh(mask, doseplane, maxdose, dd, id, structure):
    """Calculate the differential DVH for the given contour and dose plane."""
    # Multiply the structure mask by the dose plane to get the dose mask
    mask = ma.array(doseplane * dd['dosegridscaling'] * 100, mask=~mask)
    # Calculate the differential dvh
    hist, edges = np.histogram(mask.compressed(),
                               bins=maxdose,
                               range=(0, maxdose))

    # Calculate the volume for the contour for the given dose plane
    vol = sum(hist) * ((id['pixelspacing'][0]) *
                       (id['pixelspacing'][1]) *
                       (structure['thickness']))
    return hist, vol

# ========================== Test DVH Calculation =========================== #

def main():
    # Check if the app can serve a new user
    if can_serve_user():
        st.title('Dose Volume Histogram Plotter')
        st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)
        
        if 'available_data' not in st.session_state:
            st.session_state['available_data'] = False
        if 'fig' not in st.session_state:
            st.session_state['fig'] = None
        if 'data' not in st.session_state:
            st.session_state['data'] = None
        if 'data_all' not in st.session_state:
            st.session_state['data_all'] = None
            
        # Load data
        rtdose_file = st.file_uploader('Upload RTDOSE file', type=['dcm'])
        rtstruct_file = st.file_uploader('Upload RTSTRUCTURE file', type=['dcm'])
    
        if rtdose_file is not None and rtstruct_file is not None:
            if '.dcm' in rtdose_file.name and '.dcm' in rtstruct_file.name:
                try:
                # Check if a previous temporary file exists
                    if 'rtdose_tempfile' in st.session_state:
                        # Delete the previous temporary file
                        if os.path.isfile(st.session_state['rtdose_tempfile'].name):
                            os.remove(st.session_state['rtdose_tempfile'].name)
                    if 'rtstruct_tempfile' in st.session_state:
                        # Delete the previous temporary file
                        if os.path.isfile(st.session_state['rtstruct_tempfile'].name):
                            os.remove(st.session_state['rtstruct_tempfile'].name)
                        
                    # Create a new temporary file with a unique name
                    with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as rtdose_tempfile:
                        rtdose_tempfile.write(rtdose_file.read())
                        rtdosefile = rtdose_tempfile.name
                    with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as rtstruct_tempfile:
                        rtstruct_tempfile.write(rtstruct_file.read())
                        rtssfile = rtstruct_tempfile.name
        
                    # Store the filename and tempfile in session state
                    st.session_state['rtdose_tempfile'] = rtdose_tempfile
                    st.session_state['rtstruct_tempfile'] = rtstruct_tempfile
                    st.session_state['valid_data'] = True
                except Exception as e:
                    st.write(f"Error: {e}")
                    st.session_state['valid_data'] = False
        else:
            st.write('Please enter valid dicom files:')
            st.session_state['valid_data'] = False
        
        if st.session_state['valid_data']:
            RTss = dicomparser.DicomParser(rtssfile)

            RTstructures = RTss.GetStructures()
            roi_names = [structure['name'] for _,structure in RTstructures.items()]
            
            st.subheader('Choose structure')
            
            all_rois = st.radio('All ROIs?', options=['Yes', 'No'], index=1)
            
            if all_rois == "No":
                selected_rois = st.multiselect('Select ROIs', options=roi_names, default=[])
            else:
                selected_rois = st.multiselect('Select ROIs', options=roi_names, default=roi_names)
                
            btn = st.button('Calcualte DVH')
    
            if btn:
                btn = False
                if len(selected_rois) > 0:
                    # Generate the calculated DVHs
                    calcdvhs = {}
                    data = {}
                    data_all = {}
                    
                    data['ROI'] = []
                    data['mindose'] = []
                    data['maxdose'] = []
                    data['meandose'] = []
                    
                    vol_numbers = list(range(5, 75, 5)) 
                    dose_numbers = [2] + list(range(5, 100, 5)) + [97, 98, 99] 
                    vol_numbers_str = ['V'+str(vol) for vol in vol_numbers] # turn into V5 etc.
                    dose_numbers_str = ['D'+str(d) for d in dose_numbers]
                    
                    for vol in vol_numbers_str:
                        data[vol] = []
                    for dose in dose_numbers_str:
                        data[dose] = []
                    
                    fig, ax = plt.subplots()
                    for key, structure in RTstructures.items():
                        if structure['name'] in selected_rois:
                            calcdvhs[key] = dvhcalc.get_dvh(rtssfile, rtdosefile, key)
                            if (key in calcdvhs) and (len(calcdvhs[key].counts) and calcdvhs[key].counts[0]!=0):
                                st.write('DVH found for ' + structure['name'])
                                
                                sns.lineplot(y=calcdvhs[key].bins[1:], x=calcdvhs[key].counts * 100 / calcdvhs[key].counts[0], 
                                     color=dvhcalc.np.array(structure['color'], dtype=float) / 255, 
                                     label=structure['name'], linestyle='dashed', ax=ax)
                                
                                df = pd.DataFrame(
                                    {'dose': calcdvhs[key].bins[1:],
                                     'volume_percentage': calcdvhs[key].counts * 100 / calcdvhs[key].counts[0]
                                     }
                                    )

                                data_all[structure['name']] = {
                                    'dose': calcdvhs[key].bins[1:],
                                    'volper': calcdvhs[key].counts * 100 / calcdvhs[key].counts[0],
                                    'vol': calcdvhs[key].counts,
                                }
                                # Calculate statistics
                                data['ROI'].append(structure['name'])
                                data['mindose'].append(df['dose'].min())
                                data['maxdose'].append(df['dose'].max())
                                data['meandose'].append(df['dose'].mean())
                                
                                for vol, vol_str in zip(vol_numbers, vol_numbers_str):
                                    data[vol_str].append(df[df['volume_percentage'] >= vol]['dose'].max())
                                for dose, dose_str in zip(dose_numbers, dose_numbers_str):
                                    data[dose_str].append(df[df['volume_percentage'] <= dose]['dose'].min())
          
                            ax.set_title('DVH curves')
                            ax.set_ylabel('Dose (Gy)')
                            ax.set_xlabel('Volume (%)')

                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                    st.session_state['fig'] = fig
                    st.session_state['data'] = data
                    st.session_state['data_all'] = data_all
                    
            if st.session_state['fig'] is not None and st.session_state['data'] is not None:
                st.write(st.session_state['fig'])
                csv = pd.DataFrame(st.session_state['data'])
                csv2 = pd.DataFrame(st.session_state['data_all'])
                @st.cache_data
                def convert_df(df):
                   return df.to_csv(index=False).encode('utf-8')
                csv = convert_df(csv)
                csv2 = convert_df(csv2)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download Selected DVH Data", csv, r'test.csv', key='download-dvh')
                with col2:
                    st.download_button("Download All DVH Data", csv2, r'test.csv', key='download-dvh-all')
    else:
        # If Multiple user use the app and the resources are not enough
        st.write("Sorry, the app is currently overloaded. Please try again later.")
        
if __name__ == "__main__":
    main()