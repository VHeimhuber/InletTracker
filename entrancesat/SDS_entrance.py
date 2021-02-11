"""This module contains all the functions needed for detecting intermittent estuary entrance states

   Author: Valentin Heimhuber, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt

# image processing modules
import skimage.transform as transform
from skimage.graph import route_through_array
from skimage import draw
import skimage.filters as filters

# machine learning modules
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
    
# other modules
from matplotlib import gridspec
from matplotlib.pyplot import cm
import matplotlib
import pickle
import geopandas as gpd
import pandas as pd
import random



import seaborn
from shapely import geometry
from shapely.ops import nearest_points #, snap
import scipy
from datetime import datetime, timedelta
import pytz
import re

# own modules
from coastsat import SDS_tools, SDS_preprocess

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans


###################################################################################################
# GEO PROCESSING FUNCTIONS
###################################################################################################

def polygon2mask(image_shape, polygon):
    """Compute a mask from polygon.
    from github since it wasn't included in the coastsat version of scikit yet
    Parameters
    ----------
    image_shape : tuple of size 2.
        The shape of the mask.
    polygon : array_like.
        The polygon coordinates of shape (N, 2) where N is
        the number of points.
    Returns
    -------
    mask : 2-D ndarray of type 'bool'.
        The mask that corresponds to the input polygon.
    Notes
    -----
    This function does not do any border checking, so that all
    the vertices need to be within the given shape.
    Examples
    --------
    >>> image_shape = (128, 128)
    >>> polygon = np.array([[60, 100], [100, 40], [40, 40]])
    >>> mask = polygon2mask(image_shape, polygon)
    >>> mask.shape
    (128, 128)
    """
    polygon = np.asarray(polygon)
    vertex_row_coords, vertex_col_coords = polygon.T
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, image_shape)
    mask = np.zeros(image_shape, dtype=int)
    mask[fill_row_coords, fill_col_coords] = 9999
    
    return mask

def maskimage_frompolygon(image, polygonndarray):
    """
    function that uses an nparray image and an nparray polygon as input and 
    returns a copy of the image with the pixels outside the polygon masked as np.NAN
    mask an nparray image with 1 dimension based on a polygon in nparray format
    """
    image_shape = (image.shape)
    #swap x and y coordinates
    polygonndarray_conv = np.copy(polygonndarray).astype(int)
    polygonndarray_conv[:,[0, 1]] = polygonndarray[:,[1, 0]]
    mask = polygon2mask(image_shape, polygonndarray_conv) #create a mask where valid values inside the polygon are 9999
    image_masked = np.copy(image)
    image_masked[mask != 9999] = np.NAN
    return image_masked

def maskimage_frompolygon_set_value(image, polygonndarray, mask_value):
    """
    function that uses an nparray image and an nparray polygon as input and 
    returns a copy of the image with the pixels outside the polygon masked as mask_value
    mask an nparray image with 1 dimension based on a polygon in nparray format
    """
    image_shape = (image.shape)
    #swap x and y coordinates
    polygonndarray_conv = np.copy(polygonndarray).astype(int)
    polygonndarray_conv[:,[0, 1]] = polygonndarray[:,[1, 0]]
    mask = polygon2mask(image_shape, polygonndarray_conv) #create a mask where valid values inside the polygon are 9999
    image_masked = np.copy(image)
    image_masked[mask != 9999] = mask_value
    
    return image_masked


#generate bounding box in pixel coordinates
def get_bounding_box_minmax(polygonndarray):
    Xmin = np.min(polygonndarray.astype(int)[:,0])
    Xmax = np.max(polygonndarray.astype(int)[:,0])
    Ymin = np.min(polygonndarray.astype(int)[:,1])
    Ymax = np.max(polygonndarray.astype(int)[:,1])
    
    return Xmin,Xmax, Ymin,Ymax

def load_shapes_as_ndarrays(layers,Site_shps, satname, sitename, shapefile_EPSG,  georef, metadata, image_epsg):
    shapes = dict.fromkeys(layers)
    for key in shapes.keys():
        coords = []
        for b in Site_shps.loc[(Site_shps.layer==key)].geometry.exterior:
            coords = np.dstack(b.coords.xy).tolist()
            coords.append(*coords) 
        coords = SDS_tools.convert_epsg(coords, shapefile_EPSG, image_epsg)
        coords = SDS_tools.convert_world2pix(coords[0][:,:-1], georef)
        shapes[key] = coords      
    return shapes

def classify_binary_otsu(im_1d, cloud_mask):
    """
    classify a greyscale image using otsu thresholding
    returns classified image where 0 = water, 1 = dryland
    """
    vec_ndwi = im_1d.reshape(im_1d.shape[0] * im_1d.shape[1])
    vec_mask = cloud_mask.reshape(cloud_mask.shape[0] * cloud_mask.shape[1])
    vec = vec_ndwi[~vec_mask]
    # apply otsu's threshold
    vec = vec[~np.isnan(vec)]
    t_otsu = filters.threshold_otsu(vec)
    # compute classified image
    im_class = np.copy(im_1d)
    im_class[im_1d < t_otsu] = 0
    im_class[im_1d >= t_otsu] = 1
    im_class[im_1d ==np.NAN] = np.NAN
    
    return im_class, t_otsu


###################################################################################################
# IOCE entrance state detection functions
###################################################################################################

def set_openvsclosed(im_ms, inputs,jpg_out_path, cloud_mask,  georef,
                   settings_training, date, satname, Xmin, Xmax, Ymin, Ymax, image_nr, filenames_itm):
    """
    Shows the image to the user for visual detection of the entrance state. The user can select "open", "closed",
    "unclear" or "skip" if the image is of poor quality.

    VH WRL 2020

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_labels: np.array
            3D image containing a boolean image for each class in the order (sand, swash, water)
        shoreline: np.array
            array of points with the X and Y coordinates of the shoreline
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        settings: dict
            contains the following fields:
        date: string
            date at which the image was taken
        satname: string
            indicates the satname (L5,L7,L8 or S2)

    Returns:    
    -----------
        vis_open_vs_closed, 
            open, closed, unclear or skip
        skip_image
            True if image is to be excluded from analysis
        keep_checking_inloop
            True if user did not click esc
    """
    keep_checking_inloop = 'True'
    im_mNDWI = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    im_NDWI = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    if plt.get_fignums():
        # get open figure if it exists
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([12.53, 9.3])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        # according to the image shape, decide whether it is better to have the images 
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2*im_RGB.shape[0]:
            # vertical subplots
            gs = gridspec.GridSpec(3, 1)
            gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0])
            ax3 = fig.add_subplot(gs[2,0])
        else:
            # horizontal subplots
            gs = gridspec.GridSpec(1, 3)
            gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax3 = fig.add_subplot(gs[0,2])

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)  

    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    #ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax1.axis('off')
    ax1.set_xlim(Xmin-30, Xmax+30)
    ax1.set_ylim( Ymax+30, Ymin-30)  
    ax1.set_title(inputs['sitename'], fontweight='bold', fontsize=16)

    # create image 2 (classification)
    #ax2.imshow(im_class)
    ax2.imshow(im_NDWI, cmap='bwr', vmin=-1, vmax=1)
    #ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax2.axis('off')
    ax2.set_xlim(Xmin-30, Xmax+30)
    ax2.set_ylim( Ymax+30, Ymin-30)  
    ax2.set_title( 'NDWI ' + date, fontweight='bold', fontsize=16)

    # create image 3 (MNDWI)
    ax3.imshow(im_mNDWI, cmap='bwr', vmin=-1, vmax=1)
    #ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax3.axis('off')
    #plt.colorbar()
    ax3.set_xlim(Xmin-30, Xmax+30)
    ax3.set_ylim( Ymax+30, Ymin-30)  
    ax3.set_title('mNDWI ' +  satname + ' ' + str(int(((image_nr+1)/len(filenames_itm))*100)) + '%', fontweight='bold', fontsize=16)

    # if check_detection is True, let user manually accept/reject the images
    skip_image = False
    vis_open_vs_closed = 'NA'
            
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
    key_event = {}
    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key
    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    while True:
        btn_open = plt.text(1.1, 0.95, 'open ⇨', size=12, ha="right", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_closed = plt.text(-0.1, 0.95, '⇦ closed', size=12, ha="left", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_skip = plt.text(0.5, 0.95, '⇧ skip', size=12, ha="center", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w')) 
        btn_esc = plt.text(0.5, 0.07, '⇓ unclear', size=12, ha="center", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_esc = plt.text(0.5, -0.03, 'esc', size=12, ha="center", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        plt.draw()
        fig.canvas.mpl_connect('key_press_event', press)
        plt.waitforbuttonpress()
        # after button is pressed, remove the buttons
        btn_open.remove()
        btn_closed.remove()
        btn_skip.remove()
        btn_esc.remove()
        
        # keep/skip image according to the pressed key, 'escape' to break the loop
        if key_event.get('pressed') == 'right':
            skip_image = False
            vis_open_vs_closed = 'open'
            break
        elif key_event.get('pressed') == 'left':
            skip_image = False
            vis_open_vs_closed = 'closed'
            break
        elif key_event.get('pressed') == 'down':
            vis_open_vs_closed = 'unclear'
            skip_image = False
            break
        elif key_event.get('pressed') == 'up':
            vis_open_vs_closed = 'poorquality'
            skip_image = True
            break
        elif key_event.get('pressed') == 'escape':
            plt.close()
            skip_image = True
            vis_open_vs_closed = 'exit on image'
            keep_checking_inloop = 'False'
            break
            #raise StopIteration('User cancelled checking shoreline detection')
        else:
            plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings_training['save_figure'] and not skip_image:
        fig.savefig(os.path.join(jpg_out_path, date + '_' + satname + '_' + vis_open_vs_closed + '.jpg'), dpi=200)

    # Don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()
    
    return vis_open_vs_closed, skip_image, keep_checking_inloop
 



def create_training_data(metadata, settings, settings_training):
    """
    Function that lets user visually inspect satellite images and decide if 
    entrance is open or closed.
    
    This can be done for the entire dataset or to a limited number of images, which will then be used to train the machine learning classifier for open vs. closed

    VH WRL 2020

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded

        settings: dict
            contains the following fields:
        sitename: str
            String containig the name of the site
        cloud_mask_issue: boolean
            True if there is an issue with the cloud mask and sand pixels are being masked on the images
        check_detection: boolean
            True to show each invidual satellite image and let the user decide if the entrance was open or closed
    Returns:
    -----------
        output: dict
            contains the training data set for all inspected images

    """      
        
    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    
    print('Generating traning data for entrance state at: ' + sitename)
    print('Manually inspect each image to create training data. Press esc once a satisfactory number of images was inspected') 

    # initialise output data structure
    Training={}
       
    # create a subfolder to store the .jpg images showing the detection + csv file of the generated training dataset
    csv_out_path = os.path.join(filepath_data, sitename, 'User_validation_data')
    if not os.path.exists(csv_out_path):
            os.makedirs(csv_out_path)   
    jpg_out_path =  os.path.join(filepath_data, sitename, 'User_validation_data' , settings_training['username'] + '_jpg_files')     
    if not os.path.exists(jpg_out_path):      
        os.makedirs(jpg_out_path)
    
    # close all open figures
    plt.close('all')
    
    # loop through the user selecte satellites 
    for satname in settings['inputs']['sat_list']:
      
        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']      

        #randomize the time step to create a more independent training data set
        epsg_dict = dict(zip(filenames, metadata[satname]['epsg']))
        if settings_training['shuffle_training_imgs']==True:
            filenames = random.sample(filenames, len(filenames))           
        
        ##########################################
        #loop through all images and store results in pd DataFrame
        ##########################################   
        plt.close()
        keep_checking = 'True'
        for i in range(len(filenames)):
            if keep_checking == 'True':
                print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')
                
                # get image filename
                fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
                date = filenames[i][:19]
        
                # preprocess image (cloud mask + pansharpening/downsampling)
                im_ms, georef, cloud_mask, im_extra, imQA, im_nodata  = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            
                # calculate cloud cover
                cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                        (cloud_mask.shape[0]*cloud_mask.shape[1]))
                
                #skip image if cloud cover is above threshold
                if cloud_cover > settings['cloud_thresh']:     #####!!!!!##### Intermediate
                    continue
                
                #load boundary shapefiles for each scene and reproject according to satellite image epsg  
                shapes = SDS_tools.load_shapes_as_ndarrays_2(settings['inputs']['location_shps']['layer'].values, settings['inputs']['location_shps'], satname, sitename, settings['shapefile_EPSG'],
                               georef, metadata, epsg_dict[filenames[i]] ) 
                         
                #get the min and max corner (in pixel coordinates) of the entrance area that will be used for plotting the data for visual inspection
                Xmin,Xmax,Ymin,Ymax = SDS_tools.get_bounding_box_minmax(shapes['A-B Mask'])      
                    
                #Manually check entrance state to generate training data
                if settings_training['save_figure']:
                    vis_open_vs_closed, skip_image, keep_checking = set_openvsclosed(im_ms, settings['inputs'],jpg_out_path, cloud_mask, georef, settings_training, date,
                                                                                                   satname, Xmin, Xmax, Ymin, Ymax, i, filenames)     
                #add results to intermediate list
                Training[date] =  satname, vis_open_vs_closed, skip_image
    
    Training_df= pd.DataFrame(Training).transpose()
    Training_df.columns = ['satname', 'Entrance_state','skip image']
    #if len(Training_df.index) > 5:
    Training_df.to_csv(os.path.join(csv_out_path, sitename +'_visual_training_data_by_' + settings_training['username'] + '.csv'))
    
    return Training_df  




def automated_entrance_paths(metadata, settings, settings_entrance, tides_df , sat_tides_df):
    """
    Function that automatically finds the connecting path from the ocean seed point
    to the entrance receiver point, regardless weather the entrance is open or not
    
    This can be done for the entire dataset or to a limited number of images
    
    It is mainly for research and algorithm refinement purposes

    VH WRL 2020

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded
        settings: dict
            contains the following fields:
        sitename: str
            String containig the name of the site
        ndwi_whitewhater_delta: float
            Float number by which the actual NDWI values are adjusted for pixels that are classified as whitewater by the NN classifier
        ndwi_sand_delta: float
            Float number by which the actual NDWI values are adjusted for pixels that are classified as sand by the NN classifier
        path_index:
            Currently either NDWI or mNDWI: This is the spectral index that will be used for detecting the least cost path. 
        tide_bool:
            Include analysis of the tide via pyfes global tide model - need to discuss whether to leave this included or not due to user difficulties
    Returns:
    -----------
        output: pandas dataframe withthe NDWI values along the transect for each image
        if plotbool = True: each detection will be output as a plot in png as well

    """           
    #plot font size and type
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : settings_entrance['fontsize']}
    matplotlib.rc('font', **font)  
    labelsize = settings_entrance['labelsize']

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    
    # create a subfolder to store the .jpg images showing the detection + csv file of the generated training dataset
    csv_out_path = os.path.join(filepath_data, sitename,  'results_' + settings['inputs']['analysis_vrs'], 'XB' + str(settings_entrance['XB_cost_raster_amp_exponent']) + 
                                '_AB' + str(settings_entrance['AB_cost_raster_amp_exponent']) + '_' + settings_entrance['path_index'])
    if not os.path.exists(csv_out_path):
            os.makedirs(csv_out_path)  
            
    image_out_path = os.path.join(csv_out_path, 'auto_transects')
    if not os.path.exists(image_out_path):
            os.makedirs(image_out_path) 
    
    # close all open figures
    plt.close('all')   
    
    # initialise output data structure and set filenames/paths
    gdf_all = gpd.GeoDataFrame() #geopandas dataframe to store transect polylines which can be exported as a shape file
    XS={}       #dictionary to store transect data to be exported as csv or pandas dataframe
    XS_dict_fn = os.path.join(csv_out_path, sitename + '_entrance_lines_auto_' +
                                 settings_entrance['path_index'] +'_based_Loop_dict.pkl')   
    gdf_all_fn = os.path.join(csv_out_path, sitename + '_entrance_lines_auto_' +
                             settings_entrance['path_index'] +'_based_Loop_gdf.pkl')
                
    # loop through the user selected satellites 
    for satname in settings['inputs']['sat_list']:
        print('Auto generating entrance paths at ' + sitename + ' for ' + satname)
      
        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']
        
        # create epsg and dates dictionary to avoid any mixups
        epsg_dict = dict(zip(filenames, metadata[satname]['epsg']))
        dates_dict = dict(zip(filenames, metadata[satname]['dates']))
        
        #subset the tide df to the time period of the satellite
        tides_df_ss = tides_df[metadata[satname]['dates'][1]- timedelta(days=200):metadata[satname]['dates'][-1]+ timedelta(days=200)]
        sat_tides_df_ss = sat_tides_df[sat_tides_df.fn.str.contains('_'+satname+'_')]  
        
        #loop through images and process automatically
        dt0 = pytz.utc.localize(datetime(1,1,1))
        for i in range(min(len(filenames), settings_entrance['number_of_images'])):
            
            print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')
            
            # skip duplicates
            time_delta = 5*60 # 5 minutes in seconds
            dt = pytz.utc.localize(datetime.strptime(filenames[i][:19],'%Y-%m-%d-%H-%M-%S'))
            if (dt - dt0).total_seconds() < time_delta:
                print('skipping %s'%filenames[i])
                continue
            dt0 = dt
            
            
            if i < settings_entrance['skip_img_' + satname][0]:
                continue
            if i in settings_entrance['skip_img_'+ satname]:
                continue
            print(str(i) + ' ' + filenames[i])
            # read image
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            
            full_img_cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                             (cloud_mask.shape[0]*cloud_mask.shape[1]))
            if full_img_cloud_cover == 1:
                print(str(i) +  ' 100% clouds or S2 nodata issue - skipped')
                continue
            
            #tide at time of image acquisition as string
            img_tide = str(np.round(sat_tides_df_ss['tide_level'][sat_tides_df_ss['fn']==filenames[i]].values[0],2))
      
            #load all shape and area polygons in pixel coordinates to set up the configuration for the spatial analysis of entrances
            shapes = load_shapes_as_ndarrays(settings['inputs']['location_shps']['layer'].values, settings['inputs']['location_shps'], satname, sitename, settings['shapefile_EPSG'],
                                               georef, metadata, epsg_dict[filenames[i]] )   
            
            #get the min and max corner (in pixel coordinates) of the entrance area that will be used for plotting the data for visual inspection
            Xmin,Xmax,Ymin,Ymax = get_bounding_box_minmax(shapes['A-B Mask']) 
            
            # define seed and receiver points
            startIndexX, startIndexY = shapes['A'][1,:]
            stopIndexX, stopIndexY = shapes['B'][1,:]
            startIndexX_B, startIndexY_B = shapes['C'][1,:]
            stopIndexX_B, stopIndexY_B = shapes['D'][1,:]
            
            if settings_entrance['cloud_cover_ROIonly']:
                # calculate cloud cover over entrance area only
                #create a ref mask that's true everywhere 
                ref_mask = np.ones(cloud_mask.shape, dtype=bool)      
                ref_mask = maskimage_frompolygon_set_value(ref_mask, shapes['A-B Mask'], False)
                #set all cloud mask pixels outside of the entrance bounding box to False = not cloudy
                cloud_mask1 = maskimage_frompolygon_set_value(cloud_mask, shapes['A-B Mask'], False)
                cloud_cover = np.divide(sum(sum(cloud_mask1.astype(int))), sum(sum(ref_mask.astype(int))))            
            else:
                # calculate cloud cover over entire image
                cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                        (cloud_mask.shape[0]*cloud_mask.shape[1]))
                
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh']:
                print(str(i) + ' too cloudy - skipped')
                continue
        
            #if image isn't too cloudy, load loop dictionary to append data to 
            if not os.path.exists(XS_dict_fn):
                outfile = open(XS_dict_fn,'wb')
                pickle.dump(XS,outfile)
                outfile.close() #closes the pkl file but keeps the XS_dict_fn open             
            #open the output dict from pkl file
            infile = open(XS_dict_fn,'rb')
            XS = pickle.load(infile)
            infile.close() 

            #test if image was already processed and move to next image if that is the case                
            if str(dates_dict[filenames[i]].date()) + '_' + satname + '_' + settings_entrance['path_index'] + '_XB_' + img_tide in XS:
                print(str(i) + ' was already processed and is therefore skipped') 
                continue
            
            #print(str(i) + ' processing entrance for ' + settings_entrance['path_index'] ) 
                            
            # define the costsurfaceArray for across berm (X) and along berm (AB) analysis direction
            if settings_entrance['path_index'] == 'mndwi':
                costSurfaceArray  = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                costSurfaceArray_B = np.copy(costSurfaceArray)          
            if settings_entrance['path_index'] == 'ndwi':
                costSurfaceArray  = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
                costSurfaceArray_B = np.copy(costSurfaceArray)
            if settings_entrance['path_index'] == 'swir':
                costSurfaceArray = im_ms[:,:,4]
                costSurfaceArray[cloud_mask] = np.nan
                costSurfaceArray_B = im_ms[:,:,4]
                costSurfaceArray_B[cloud_mask] = np.nan
            if settings_entrance['path_index'] == 'nir':
                costSurfaceArray = im_ms[:,:,3]
                costSurfaceArray[cloud_mask] = np.nan
                costSurfaceArray_B = im_ms[:,:,3]
                costSurfaceArray_B[cloud_mask] = np.nan
            if settings_entrance['path_index'] == 'index':
#                costSurfaceArray = im_ms[:,:,1] / im_ms[:,:,0]
#                costSurfaceArray[cloud_mask] = np.nan
#                costSurfaceArray_B = im_ms[:,:,4]
#                costSurfaceArray_B[cloud_mask] = np.nan    
                #use mNDWI for path finding to get a transect along the deepest part of lake
                costSurfaceArray  = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                costSurfaceArray_B = np.copy(costSurfaceArray)
                
            #finalize the costsurface array to facilitate better path finding for inlet situations
            # XB: the routthrougharray algorithm did not work well with negative values so spectral indices are shifted into the positive here via +1
            costSurfaceArray_A = costSurfaceArray + 2    #shift the costsurface to above 1 so that we can do proper exponentiation
            costSurfaceArray_A = maskimage_frompolygon_set_value(costSurfaceArray_A, shapes['A-B Mask'], 1000)          
            costSurfaceArray_A[np.isnan(costSurfaceArray_A)] = 100
            costSurfaceArray_A = np.power(costSurfaceArray_A, settings_entrance['XB_cost_raster_amp_exponent'])
            
            # AB: for along_berm analysis, we want the 'driest' path over the beach berm so we need to invert the costsurface via -1 first before shifting via + 2 so that min =1 for exponentiation
            if settings_entrance['path_index'] != 'index':
                costSurfaceArray_B = (costSurfaceArray_B * (-1)) + 3
            else: 
                costSurfaceArray_B = costSurfaceArray + 2 
                
            if settings_entrance['use_berm_mask_for_AB']:
                costSurfaceArray_B = maskimage_frompolygon_set_value(costSurfaceArray_B, shapes['C-D Mask'], 1000)   
            else:
                costSurfaceArray_B = maskimage_frompolygon_set_value(costSurfaceArray_B, shapes['A-B Mask'], 1000)                
            costSurfaceArray_B[np.isnan(costSurfaceArray_B)] = 1000
            costSurfaceArray_B = np.power(costSurfaceArray_B, settings_entrance['AB_cost_raster_amp_exponent'])

            #execute the least cost path search algorithm in AB direction
            try:
                indices_B, weight_B = route_through_array(costSurfaceArray_B, (int(startIndexY_B),int(startIndexX_B)),
                                          (int(stopIndexY_B),int(stopIndexX_B)),geometric=True,fully_connected=True)
            except:
                print(str(i) +  ' no path could be found - likely due to excessive cloud or L7 SLE')
                continue
            
            #invert the x y values to be in line with the np image array conventions used in coastsat        
            indices_B = list(map(lambda sub: (sub[1], sub[0]), indices_B))
            indices_B = np.array(indices_B)
            
            #create indexed raster from indices
            path_B = np.zeros_like(costSurfaceArray)
            path_B[indices_B.T[1], indices_B.T[0]] = 1
           
            # geospatial processing of the least cost path including coordinate transformations and splitting the path into intervals of 1m
            geoms_B = []
            # convert pixel coordinates to world coordinates
            pts_world_B = SDS_tools.convert_pix2world(indices_B[:,[1,0]], georef)
            #interpolate between array incices to account for different distances across diagonal vs orthogonal pixel paths
            pts_world_interp_B = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
            for k in range(len(pts_world_B)-1):
                #create a path between subsequent pairs of pixel centrepoints into intervals of 1m
                pt_dist = np.linalg.norm(pts_world_B[k,:]-pts_world_B[k+1,:])
                xvals = np.arange(0,pt_dist)
                yvals = np.zeros(len(xvals))
                pt_coords = np.zeros((len(xvals),2))
                pt_coords[:,0] = xvals
                pt_coords[:,1] = yvals
                phi = 0
                deltax = pts_world_B[k+1,0] - pts_world_B[k,0]
                deltay = pts_world_B[k+1,1] - pts_world_B[k,1]
                phi = np.pi/2 - np.math.atan2(deltax, deltay)
                tf = transform.EuclideanTransform(rotation=phi, translation=pts_world_B[k,:])
                pts_world_interp_B = np.append(pts_world_interp_B,tf(pt_coords), axis=0)
            pts_world_interp_B = np.delete(pts_world_interp_B,0,axis=0)
            # convert world image coordinates to user-defined coordinate system
            pts_world_interp_reproj_B = SDS_tools.convert_epsg(pts_world_interp_B,  epsg_dict[filenames[i]], settings['output_epsg'])
            pts_pix_interp_B = SDS_tools.convert_world2pix(pts_world_interp_B, georef)
            #save as geometry (to create .geojson file later)
            geoms_B.append(geometry.LineString(pts_world_interp_reproj_B))        
            
            #extract spectral indices along the digitized line.
            im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)           
            im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
            im_swir = im_ms[:,:,4]
            im_nir = im_ms[:,:,3]
            im_swir[cloud_mask] = np.nan

            if settings_entrance['index_id'] == 'bandratio':  
                im_bathy_sdb = im_ms[:,:,settings_entrance['band1']] / im_ms[:,:,settings_entrance['band2']]
            if settings_entrance['index_id'] == 'NIR':   
                im_bathy_sdb = im_ms[:,:,3]
            if settings_entrance['index_id'] == 'ImprovedNIRwithSAC':   
                im_bathy_sdb = im_ms[:,:,3] -1.03* im_ms[:,:,4]          
            if settings_entrance['index_id'] == 'NIRoverRedwithSAC':   
                im_bathy_sdb = (im_ms[:,:,3] - im_ms[:,:,4]) / (im_ms[:,:,2] - im_ms[:,:,4]) 
            if settings_entrance['index_id'] == 'greenminusred':   
                im_bathy_sdb = im_ms[:,:,1] - (im_ms[:,:,1] / im_ms[:,:,2])            
            im_bathy_sdb[cloud_mask] = np.nan
            
            z_mndwi_B = scipy.ndimage.map_coordinates(im_mndwi, np.vstack((pts_pix_interp_B[:,1], pts_pix_interp_B[:,0])),order=1)
            z_ndwi_B = scipy.ndimage.map_coordinates(im_ndwi, np.vstack((pts_pix_interp_B[:,1], pts_pix_interp_B[:,0])),order=1)
            z_swir_B = scipy.ndimage.map_coordinates(im_swir, np.vstack((pts_pix_interp_B[:,1], pts_pix_interp_B[:,0])),order=1)
            z_nir_B = scipy.ndimage.map_coordinates(im_nir, np.vstack((pts_pix_interp_B[:,1], pts_pix_interp_B[:,0])),order=1)
            z_bathy_B = scipy.ndimage.map_coordinates(im_bathy_sdb, np.vstack((pts_pix_interp_B[:,1], pts_pix_interp_B[:,0])),order=1)
            
            #append to XS dictionary
            XS[filenames[i][:19] + '_' + satname + '_mndwi_AB_' + img_tide] = z_mndwi_B
            XS[filenames[i][:19] + '_' + satname + '_ndwi_AB_' + img_tide] = z_ndwi_B
            XS[filenames[i][:19] + '_' + satname + '_swir_AB_' + img_tide] = z_swir_B
            XS[filenames[i][:19] + '_' + satname + '_nir_AB_' + img_tide] = z_nir_B
            XS[filenames[i][:19] + '_' + satname + '_bathy_AB_' + img_tide] = z_bathy_B
                
            #Do the path finding in the across berm direction: Most of the following 
            #bits of code could are duplicate of the AB transect finding and could potentially be written more elegantly in a loop
            indices, weight = route_through_array(costSurfaceArray_A, (int(startIndexY),int(startIndexX)),
                                                  (int(stopIndexY),int(stopIndexX)),geometric=True,fully_connected=True)             
                        
            #invert the x y values to be in line with the np image array conventions used in coastsat
            indices = list(map(lambda sub: (sub[1], sub[0]), indices))
            indices = np.array(indices)
            
            #create indexed raster from indices
            path = np.zeros_like(costSurfaceArray)
            path[indices.T[1], indices.T[0]] = 1
            
            # geospatial processing of the least cost path including coordinate transformations and splitting the path into intervals of 1m
            geoms = []
            # convert pixel coordinates to world coordinates
            pts_world = SDS_tools.convert_pix2world(indices[:,[1,0]], georef)
            #interpolate between array incices to account for different distances across diagonal vs orthogonal pixel paths
            pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
            for k in range(len(pts_world)-1):
                #create a path between subsequent pairs of pixel centrepoints into intervals of 1m
                pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
                xvals = np.arange(0,pt_dist)
                yvals = np.zeros(len(xvals))
                pt_coords = np.zeros((len(xvals),2))
                pt_coords[:,0] = xvals
                pt_coords[:,1] = yvals
                phi = 0
                deltax = pts_world[k+1,0] - pts_world[k,0]
                deltay = pts_world[k+1,1] - pts_world[k,1]
                phi = np.pi/2 - np.math.atan2(deltax, deltay)
                tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
                pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0)
            pts_world_interp = np.delete(pts_world_interp,0,axis=0)
            # convert world image coordinates to user-defined coordinate system
            pts_world_interp_reproj = SDS_tools.convert_epsg(pts_world_interp,  epsg_dict[filenames[i]], settings['output_epsg'])
            pts_pix_interp = SDS_tools.convert_world2pix(pts_world_interp, georef)
            #save as geometry (to create .geojson file later)
            geoms.append(geometry.LineString(pts_world_interp_reproj))
            
            #write XB indices transects to dictionary 
            z_mndwi = scipy.ndimage.map_coordinates(im_mndwi, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
            z_ndwi = scipy.ndimage.map_coordinates(im_ndwi, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
            z_swir = scipy.ndimage.map_coordinates(im_swir, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)  
            z_nir = scipy.ndimage.map_coordinates(im_nir, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1) 
            z_bathy = scipy.ndimage.map_coordinates(im_bathy_sdb, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
            XS[filenames[i][:19] + '_' + satname + '_mndwi_XB_' + img_tide] = z_mndwi
            XS[filenames[i][:19] + '_' + satname + '_ndwi_XB_' + img_tide] = z_ndwi
            XS[filenames[i][:19] + '_' + satname + '_swir_XB_' + img_tide] = z_swir
            XS[filenames[i][:19] + '_' + satname + '_nir_XB_' + img_tide] = z_nir
            XS[filenames[i][:19] + '_' + satname + '_bathy_XB_' + img_tide] = z_bathy         
                       
            # Find and located intersection between transects: https://github.com/Toblerity/Shapely/issues/190                              
            XBline = geometry.LineString(pts_pix_interp)
            ABline = geometry.LineString(pts_pix_interp_B)
            XBline_mp = geometry.MultiPoint(pts_pix_interp)
            ABline_mp = geometry.MultiPoint(pts_pix_interp_B)            
            #calculate the exact point of intersection (could be between vertices as this is done based on polylines)
            intersection  = ABline.intersection(XBline)        
            if intersection.is_empty:
                print('no intersection exists between XB and AB transects')
                Intersection_coords = []             
            else: 
                try:
                    intersection  = intersection[0]
                except:
                    #print('intersection was already a point')
                    print('')
                Intersection_coords = [intersection.coords[:][0][0], intersection.coords[:][0][1]] 
                #this throws an error when the intersection returns multiple points or a point and a linestring 
                #Find nearest point in the multipoint versions of the transects and the intersection
                AB_nearest_vertice = nearest_points(ABline_mp, intersection)[0].coords[:][0]
                AB_nearest_vertice =  [AB_nearest_vertice[0], AB_nearest_vertice[1]]
                AB_distance_to_intersection =  [list(item) for item in pts_pix_interp_B].index(AB_nearest_vertice)               
                XB_nearest_vertice = nearest_points(XBline_mp, intersection)[0].coords[:][0]
                XB_nearest_vertice =  [XB_nearest_vertice[0], XB_nearest_vertice[1]]
                XB_distance_to_intersection =  [list(item) for item in pts_pix_interp].index(XB_nearest_vertice)             
                XS[filenames[i][:19]  + '_' + satname + '_AB_distance_to_intersection'] = AB_distance_to_intersection  
                XS[filenames[i][:19]  + '_' + satname + '_XB_distance_to_intersection'] = XB_distance_to_intersection            
                
            #dump the XS dictionary as pkl file again
            outfile = open(XS_dict_fn,'wb')
            pickle.dump(XS,outfile)
            outfile.close()
        
            # also store as .geojson in case user wants to drag-and-drop on GIS for verification
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms))
            gdf.index = [k]
            gdf.loc[k,'name'] = 'entrance_line_XB_' + str(k+1)
            gdf.loc[k,'date'] = filenames[i][:19]
            gdf.loc[k,'satname'] = satname
            gdf.loc[k,'direction'] = 'XB'
            
            gdf_B = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms_B))
            gdf_B.index = [k]
            gdf_B.loc[k,'name'] = 'entrance_line_AB_' + str(k+1)
            gdf_B.loc[k,'date'] = filenames[i][:19]
            gdf_B.loc[k,'satname'] = satname
            gdf_B.loc[k,'direction'] = 'AB'
            
            # store into geodataframe and dump as pkl file
            #load or create pkl file 
            if not os.path.exists(gdf_all_fn): 
                outfile = open(gdf_all_fn,'wb')
                pickle.dump(gdf_all,outfile)
                outfile.close()
                
            infile = open(gdf_all_fn,'rb')
            gdf_all = pickle.load(infile)
            infile.close()  
            
            #append new transects
            gdf_all = gdf_all.append(gdf)
            gdf_all = gdf_all.append(gdf_B)
            
            #dump as pkl file   
            outfile = open(gdf_all_fn,'wb')
            pickle.dump(gdf_all,outfile)
            outfile.close() 
                  
            if settings_entrance['plot_bool']:                          
                #setup the figure
                fig = plt.figure(figsize=(40,30)) 
                
                #plot RGB image
                ax=plt.subplot(4,3,3)
                im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)  
                plt.imshow(im_RGB, interpolation="bicubic") 
                plt.rcParams["axes.grid"] = False
                plt.title(satname + ' ' +str(dates_dict[filenames[i]].date()), fontsize=settings_entrance['axlabelsize'])
                ax.axis('off')
                plt.xlim(Xmin-settings_entrance['img_crop_adjsut'], Xmax+settings_entrance['img_crop_adjsut'])
                plt.ylim(Ymax,Ymin) 
                ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp[0,0]+2, pts_pix_interp[0,1]+2,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp[-1,0]+2, pts_pix_interp[-1,1]+2,'B',horizontalalignment='left', color='yellow', fontsize=16)               
                ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)                
                if len(Intersection_coords) >= 1:
                    ax.plot(Intersection_coords[0], Intersection_coords[1],"x", color='cyan', markersize=15, lw=1.5)
                    #plt.text(Intersection_coords[0]+2, Intersection_coords[1]+2,'X',horizontalalignment='left', color='cyan', fontsize=16)
                
                if settings_entrance['plot_entrance_bbx']:
                    ptsbbx_world = SDS_tools.convert_pix2world(shapes['A-B Mask'], georef)
                    #interpolate between array incices to account for different distances across diagonal vs orthogonal pixel paths
                    ptsbbx_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                    for k in range(len(ptsbbx_world)-1):
                        #create a path between subsequent pairs of pixel centrepoints into intervals of 1m
                        pt_dist = np.linalg.norm(ptsbbx_world[k,:]-ptsbbx_world[k+1,:])
                        xvals = np.arange(0,pt_dist)
                        yvals = np.zeros(len(xvals))
                        pt_coords = np.zeros((len(xvals),2))
                        pt_coords[:,0] = xvals
                        pt_coords[:,1] = yvals
                        phi = 0
                        deltax = ptsbbx_world[k+1,0] - ptsbbx_world[k,0]
                        deltay = ptsbbx_world[k+1,1] - ptsbbx_world[k,1]
                        phi = np.pi/2 - np.math.atan2(deltax, deltay)
                        tf = transform.EuclideanTransform(rotation=phi, translation=ptsbbx_world[k,:])
                        ptsbbx_world_interp = np.append(ptsbbx_world_interp,tf(pt_coords), axis=0)
                    ptsbbx_world_interp = np.delete(ptsbbx_world_interp,0,axis=0)
                    ptsbbx_pix_interp = SDS_tools.convert_world2pix(ptsbbx_world_interp, georef)
                    ax.plot(ptsbbx_pix_interp[:,1], ptsbbx_pix_interp[:,0], 'r--', color='fuchsia')  
                if settings_entrance['use_berm_mask_for_AB']:
                    ptsbbx_world = SDS_tools.convert_pix2world(shapes['C-D Mask'], georef)
                    #interpolate between array incices to account for different distances across diagonal vs orthogonal pixel paths
                    ptsbbx_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                    for k in range(len(ptsbbx_world)-1):
                        #create a path between subsequent pairs of pixel centrepoints into intervals of 1m
                        pt_dist = np.linalg.norm(ptsbbx_world[k,:]-ptsbbx_world[k+1,:])
                        xvals = np.arange(0,pt_dist)
                        yvals = np.zeros(len(xvals))
                        pt_coords = np.zeros((len(xvals),2))
                        pt_coords[:,0] = xvals
                        pt_coords[:,1] = yvals
                        phi = 0
                        deltax = ptsbbx_world[k+1,0] - ptsbbx_world[k,0]
                        deltay = ptsbbx_world[k+1,1] - ptsbbx_world[k,1]
                        phi = np.pi/2 - np.math.atan2(deltax, deltay)
                        tf = transform.EuclideanTransform(rotation=phi, translation=ptsbbx_world[k,:])
                        ptsbbx_world_interp = np.append(ptsbbx_world_interp,tf(pt_coords), axis=0)
                    ptsbbx_world_interp = np.delete(ptsbbx_world_interp,0,axis=0)
                    ptsbbx_pix_interp = SDS_tools.convert_world2pix(ptsbbx_world_interp, georef)
                    ax.plot(ptsbbx_pix_interp[:,1], ptsbbx_pix_interp[:,0], 'r--', color='lime')  
                
                #plot tidal histogram instead of RGB if desired
                if settings_entrance['plot_tide_histogram']:
                    # plot time-step distribution
                    seaborn.kdeplot(tides_df_ss['tide_level'], shade=True,vertical=False, color='blue',bw=settings_entrance['hist_bw'],legend=False, lw=2, ax=ax)
                    seaborn.kdeplot(sat_tides_df_ss['tide_level'], shade=True,vertical=False, color='lightblue',bw=settings_entrance['hist_bw'], legend=False, lw=2, ax=ax)
                    plt.xlim(-1,1)
                    plt.ylabel('Probability density', fontsize=settings_entrance['axlabelsize'])
                    plt.xlabel('Tides over full period (darkblue) and during images only (lightblue)', fontsize=settings_entrance['axlabelsize'])
                    plt.axvline(x=sat_tides_df_ss['tide_level'][i], color='red', linestyle='dotted', lw=2, alpha=0.9) 
                    #plt.text(sat_tides_df_ss['tide_level'][i] , 0.5 ,  'tide @image', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                                   
#                    t = np.array([_.timestamp() for _ in dates_sat]).astype('float64')
#                    delta_t = np.diff(t)
#                    #fig, ax = plt.subplots(1,1,figsize=(12,3), tight_layout=True)
#                    ax.grid(which='major', linestyle=':', color='0.5')
#                    bins = np.arange(np.min(delta_t)/seconds_in_day, np.max(delta_t)/seconds_in_day+1,1)-0.5
#                    ax.hist(delta_t/seconds_in_day, bins=bins, ec='k', width=1);
#                    ax.set(xlabel='timestep [days]', ylabel='counts',
#                           xticks=n_days*np.arange(0,20),
#                           xlim=[0,50], title='Timestep distribution');
                
                #plot SWIR1
                ax=plt.subplot(4,3,1) 
                plt.imshow(im_ms[:,:,4], cmap='seismic', vmin=0, vmax=0.8) 
                if settings_entrance['path_index']== 'swir':
                    plt.title(r"$\bf{" + 'SWIR1' + "}$", fontsize=settings_entrance['axlabelsize'])
                else:
                    plt.title('SWIR1' , fontsize=settings_entrance['axlabelsize'] )
                ax.axis('off')
                plt.rcParams["axes.grid"] = False
                plt.xlim(Xmin-settings_entrance['img_crop_adjsut'], Xmax+settings_entrance['img_crop_adjsut'])
                plt.ylim(Ymax,Ymin) 
                ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp[0,0]+2, pts_pix_interp[0,1]+2,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp[-1,0]+2, pts_pix_interp[-1,1]+2,'B',horizontalalignment='left', color='yellow', fontsize=16)               
                ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)                
                if len(Intersection_coords) >= 1:
                    ax.plot(Intersection_coords[0], Intersection_coords[1],"x", color='cyan', markersize=15, lw=1.5)
                if settings_entrance['plt_colorbars']:
                    plt.colorbar()
                                  
                #plot NIR
                ax=plt.subplot(4,3,2) 
                plt.imshow(im_ms[:,:,3], cmap='seismic', vmin=0, vmax=0.8) 
                if settings_entrance['path_index']== 'nir':
                    plt.title(r"$\bf{" + 'NIR' + "}$", fontsize=settings_entrance['axlabelsize'])
                else:
                    plt.title('NIR', fontsize=settings_entrance['axlabelsize'])
                ax.axis('off')
                plt.xlim(Xmin-settings_entrance['img_crop_adjsut'], Xmax+settings_entrance['img_crop_adjsut'])
                plt.ylim(Ymax,Ymin)  
                ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp[0,0]+2, pts_pix_interp[0,1]+2,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp[-1,0]+2, pts_pix_interp[-1,1]+2,'B',horizontalalignment='left', color='yellow', fontsize=16)               
                ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)                
                if len(Intersection_coords) >= 1:
                    ax.plot(Intersection_coords[0], Intersection_coords[1],"x", color='cyan', markersize=15, lw=1.5)        
                if settings_entrance['plt_colorbars']:
                    plt.colorbar()
                    
                #plot mNDWI
                ax=plt.subplot(4,3,4) 
                plt.imshow(im_mndwi, cmap='seismic', vmin=-1, vmax=1) 
                ax.axis('off')
                plt.rcParams["axes.grid"] = False
                if settings_entrance['path_index']== 'mndwi':
                    plt.title(r"$\bf{" + 'modified NDWI' + "}$", fontsize=settings_entrance['axlabelsize'])
                else:
                    plt.title('modified NDWI' , fontsize=settings_entrance['axlabelsize'] )
                #plt.colorbar()
                plt.xlim(Xmin-settings_entrance['img_crop_adjsut'], Xmax+settings_entrance['img_crop_adjsut'])
                plt.ylim(Ymax,Ymin) 
                ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp[0,0]+1, pts_pix_interp[0,1]+1,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp[-1,0]+1, pts_pix_interp[-1,1]+1,'B',horizontalalignment='left', color='yellow', fontsize=16)
                ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)
                if settings_entrance['plt_colorbars']:
                    plt.colorbar()
                    
                #plot NDWI
                ax=plt.subplot(4,3,5) 
                plt.imshow(im_ndwi, cmap='seismic', vmin=-1, vmax=1) 
                ax.axis('off')
                if settings_entrance['path_index']== 'ndwi':
                    plt.title(r"$\bf{" + 'NDWI' + "}$", fontsize=settings_entrance['axlabelsize'])
                else:
                    plt.title('NDWI', fontsize=settings_entrance['axlabelsize'])
                #plt.colorbar()
                plt.rcParams["axes.grid"] = False
                plt.xlim(Xmin-settings_entrance['img_crop_adjsut'], Xmax+settings_entrance['img_crop_adjsut'])
                plt.ylim(Ymax,Ymin) 
                ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp[0,0]+1, pts_pix_interp[0,1]+1,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp[-1,0]+1, pts_pix_interp[-1,1]+1,'B',horizontalalignment='left', color='yellow', fontsize=16)
                ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
                ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
                ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
                plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)
                if settings_entrance['plt_colorbars']:
                    plt.colorbar()
                    
                #plot tide series or, if not chosen, the alternative index
                ax=plt.subplot(4,3,6) 
                if settings_entrance['tide_bool']:               
                    ax.set_title('Tide level at img aquisition = ' + img_tide +  ' [m aMSL]', fontsize=settings_entrance['axlabelsize'])
                    ax.grid(which='major', linestyle=':', color='0.5')
                    ax.plot(tides_df_ss.index, tides_df_ss['tide_level'], '-', color='0.6')
                    ax.plot(sat_tides_df_ss.index, sat_tides_df_ss['tide_level'], '-o', color='k', ms=4, mfc='w',lw=1)
                    plt.axhline(y=sat_tides_df_ss['tide_level'][i], color='red', linestyle='dotted', lw=2, alpha=0.9) 
                    ax.plot(sat_tides_df_ss.index[i], sat_tides_df_ss['tide_level'][i], '-o', color='red', ms=12, mfc='w',lw=7)
                    ax.set_ylabel('tide level [m]', fontsize=settings_entrance['axlabelsize'])
                    ax.set_ylim(min(tides_df_ss['tide_level']), max(tides_df_ss['tide_level']))
                    #plt.text(sat_tides_df_ss['tide_level'][i] , 0.5 ,  'tide @image', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                    
                else:
                    plt.imshow(im_bathy_sdb, cmap='seismic') #, vmin=0.9, vmax=1.1) 
                    ax.axis('off')
                    plt.title(settings_entrance['path_index_id'] + ' index ', fontsize=settings_entrance['axlabelsize'])
                    #plt.colorbar()
                    plt.rcParams["axes.grid"] = False
                    plt.xlim(Xmin-settings_entrance['img_crop_adjsut'], Xmax+settings_entrance['img_crop_adjsut'])
                    plt.ylim(Ymax,Ymin) 
                    ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
                    ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
                    ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
                    plt.text(pts_pix_interp[0,0]+1, pts_pix_interp[0,1]+1,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                    plt.text(pts_pix_interp[-1,0]+1, pts_pix_interp[-1,1]+1,'B',horizontalalignment='left', color='yellow', fontsize=16)
                    ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
                    ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
                    ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
                    plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
                    plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)
                                      
                ax=plt.subplot(4,3,7)
                if settings_entrance['path_index'] in ['ndwi', 'nir']:  
                    pd.DataFrame(z_ndwi).plot(color='blue', linestyle='--', ax=ax) 
                    plt.axhline(y=np.nanpercentile(z_ndwi_B, 50),  color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    plt.ylabel('NDWI [-]', fontsize=settings_entrance['axlabelsize'])                    
                if settings_entrance['path_index'] in ['mndwi', 'swir', 'index']: 
                    pd.DataFrame(z_mndwi).plot(color='blue', linestyle='--', ax=ax)                
                    plt.axhline(y=np.nanpercentile(z_mndwi_B, settings_entrance['sand_percentile']) , xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)   
                    plt.ylabel('mNDWI [-]', fontsize=settings_entrance['axlabelsize'])                             
                    plt.ylim(-0.9,0.9) 
                if len(Intersection_coords) >= 1:               
                    plt.axvline(x=XB_distance_to_intersection , color='cyan', linestyle='--', lw=1.4, alpha=0.85) 
                    plt.text(XB_distance_to_intersection , -0.88 ,  'X', color='cyan', rotation=90, ha='right', va='bottom',fontsize=13, alpha=settings_entrance['vhline_transparancy'])
                plt.ylim(-0.9,0.9)
                plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=2, alpha=0.9) 
                plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
                plt.text(len(z_mndwi)-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
                plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                ax.get_legend().remove()
                
                ax=plt.subplot(4,3,8)
                if settings_entrance['path_index'] in ['ndwi', 'nir']:  
                    pd.DataFrame(z_ndwi_B).plot(color='blue', linestyle='--', ax=ax) 
                    plt.axhline(y=np.nanpercentile(z_ndwi_B, 50), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    plt.ylabel('NDWI [-]', fontsize=settings_entrance['axlabelsize'])                  
                if settings_entrance['path_index'] in ['mndwi', 'swir', 'index']:  
                    pd.DataFrame(z_mndwi_B).plot(color='blue', linestyle='--', ax=ax)  
                    plt.axhline(y=np.nanpercentile(z_mndwi_B, 50), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    plt.ylabel('mNDWI [-]', fontsize=settings_entrance['axlabelsize'])
                if len(Intersection_coords) >= 1:               
                    plt.axvline(x=AB_distance_to_intersection , color='cyan', linestyle='--', lw=1.4, alpha=0.85) 
                    plt.text(AB_distance_to_intersection , -0.88 ,  'X', color='cyan', rotation=90, ha='right', va='bottom',fontsize=13, alpha=settings_entrance['vhline_transparancy'])         
                plt.ylim(-0.9,0.9)
                plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=2, alpha=0.9)        
                plt.text(1,0,'C',horizontalalignment='left', color='grey' , fontsize=labelsize)
                plt.text(len(z_mndwi_B)-2,0,'D',horizontalalignment='right', color='grey' , fontsize=labelsize)
                plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                ax.get_legend().remove()  
                
                ax=plt.subplot(4,3,9)
                if settings_entrance['path_index'] in ['mndwi', 'swir', 'index']: 
                    seaborn.kdeplot(z_mndwi, shade=True,vertical=False, color='lightblue',bw=settings_entrance['hist_bw'], legend=False, lw=2, ax=ax)
                    seaborn.kdeplot(z_mndwi_B, shade=True,vertical=False, color='orange',bw=settings_entrance['hist_bw'],legend=False, lw=2, ax=ax)
                    plt.axvline(x=np.nanpercentile(z_mndwi_B, settings_entrance['sand_percentile'] ), color='orchid', linestyle='dotted', lw=2, alpha=1) 
                    plt.text(np.nanpercentile(z_mndwi_B, settings_entrance['sand_percentile'] ) , 0.5 ,  str(settings_entrance['sand_percentile']) + 'p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                      
                    plt.xlim(-1,0.5)
                    plt.ylabel('Probability density', fontsize=settings_entrance['axlabelsize'])
                    plt.xlabel('mNDWI over A-B (lightblue) and C-D (orange) transect', fontsize=settings_entrance['axlabelsize'])
                if settings_entrance['path_index'] in ['ndwi', 'nir']:  
                    seaborn.kdeplot(z_ndwi, shade=True,vertical=False, color='lightblue',bw=settings_entrance['hist_bw'], legend=False, lw=2, ax=ax)
                    seaborn.kdeplot(z_ndwi_B, shade=True,vertical=False, color='orange',bw=settings_entrance['hist_bw'],legend=False, lw=2, ax=ax)
                    img_ndwi_perc = np.nanpercentile(z_ndwi_B, settings_entrance['sand_percentile'] )
                    plt.axvline(x=img_ndwi_perc, color='orchid', linestyle='dotted', lw=2, alpha=1) 
                    plt.text(img_ndwi_perc , 0.5 ,  str(settings_entrance['sand_percentile']) + 'p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                      
                    plt.xlim(-1,0.5)
                    plt.ylabel('Probability density', fontsize=settings_entrance['axlabelsize'])
                    plt.xlabel('NDWI over A-B (lightblue) and C-D (orange) transect', fontsize=settings_entrance['axlabelsize'])               
                
                #plot single band transects A-B
                ax=plt.subplot(4,3,10)
                if settings_entrance['path_index'] in ['ndwi', 'nir']: 
                    pd.DataFrame(z_nir).plot(color='blue', linestyle='--', ax=ax) 
                    #pd.DataFrame(z_ndwi_B_adj).plot(color='blue', linestyle='--', ax=ax) 
                    plt.ylim(0,0.7)
                    plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=2, alpha=0.9) 
                    plt.axhline(y=np.nanpercentile(z_nir_B, settings_entrance['sand_percentile']), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    if len(Intersection_coords) >= 1:               
                        plt.axvline(x=XB_distance_to_intersection , color='cyan', linestyle='--', lw=1.4, alpha=0.85) 
                        plt.text(XB_distance_to_intersection , 0.02 ,  'X', color='cyan', rotation=90, ha='right', va='bottom',fontsize=13, alpha=settings_entrance['vhline_transparancy'])         
                    plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
                    plt.text(len(z_nir)-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
                    plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                    plt.ylabel('NIR [-]', fontsize=settings_entrance['axlabelsize'])
                    ax.get_legend().remove()  
                if settings_entrance['path_index'] in ['swir', 'mndwi']:
                    pd.DataFrame(z_swir).plot(color='blue', linestyle='--', ax=ax) 
                    plt.ylim(0,0.7)
                    plt.axhline(y=np.nanpercentile(z_swir_B, settings_entrance['sand_percentile']), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    if len(Intersection_coords) >= 1:               
                        plt.axvline(x=XB_distance_to_intersection , color='cyan', linestyle='--', lw=1.4, alpha=0.85) 
                        plt.text(XB_distance_to_intersection , 0.02 ,  'X', color='cyan', rotation=90, ha='right', va='bottom', fontsize=13, alpha=settings_entrance['vhline_transparancy'])
                    plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
                    plt.text(len(z_swir)-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
                    plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                    plt.ylabel('SWIR1 [-]', fontsize=settings_entrance['axlabelsize'])
                    ax.get_legend().remove()                      
                if settings_entrance['path_index'] == 'index': 
                    pd.DataFrame(z_bathy).plot(color='blue', linestyle='--', ax=ax) 
                    #plt.ylim(0.6,1.4)
                    plt.axhline(y=np.nanpercentile(z_bathy_B, settings_entrance['sand_percentile']), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                    plt.ylabel(settings_entrance['path_index_id'] + ' Index [-]', fontsize=settings_entrance['axlabelsize'])
                    ax.get_legend().remove()
                 
                #plot single band transects C-D
                ax=plt.subplot(4,3,11)
                if settings_entrance['path_index'] in ['swir', 'mndwi']:
                    pd.DataFrame(z_swir_B).plot(color='blue', linestyle='--', ax=ax) 
                    #pd.DataFrame(z_ndwi_B_adj).plot(color='blue', linestyle='--', ax=ax) 
                    plt.ylim(0,0.7)
                    plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=2, alpha=0.9) 
                    plt.axhline(y=np.nanpercentile(z_swir_B, settings_entrance['sand_percentile']), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    if len(Intersection_coords) >= 1:               
                        plt.axvline(x=AB_distance_to_intersection , color='cyan', linestyle='--', lw=1.4, alpha=0.85) 
                        plt.text(AB_distance_to_intersection , 0.02 ,  'X', color='cyan', rotation=90, ha='right', va='bottom',fontsize=13, alpha=settings_entrance['vhline_transparancy'])         
                    plt.text(1,0,'C',horizontalalignment='left', color='grey' , fontsize=labelsize)
                    plt.text(len(z_swir_B)-2,0,'D',horizontalalignment='right', color='grey' , fontsize=labelsize)
                    plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                    plt.ylabel('SWIR1 [-]', fontsize=settings_entrance['axlabelsize'])
                    ax.get_legend().remove()
                if settings_entrance['path_index'] in ['ndwi', 'nir']: 
                    pd.DataFrame(z_nir_B).plot(color='blue', linestyle='--', ax=ax) 
                    #pd.DataFrame(z_ndwi_B_adj).plot(color='blue', linestyle='--', ax=ax) 
                    plt.ylim(0,0.7)
                    plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=2, alpha=0.9) 
                    plt.axhline(y=np.nanpercentile(z_nir_B, settings_entrance['sand_percentile']), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    if len(Intersection_coords) >= 1:               
                        plt.axvline(x=AB_distance_to_intersection , color='cyan', linestyle='--', lw=1.4, alpha=0.85) 
                        plt.text(AB_distance_to_intersection , 0.02 ,  'X', color='cyan', rotation=90, ha='right', va='bottom',fontsize=13, alpha=settings_entrance['vhline_transparancy'])         
                    plt.text(1,0,'C',horizontalalignment='left', color='grey' , fontsize=labelsize)
                    plt.text(len(z_nir_B)-2,0,'D',horizontalalignment='right', color='grey' , fontsize=labelsize)
                    plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                    plt.ylabel('NIR [-]', fontsize=settings_entrance['axlabelsize'])
                    ax.get_legend().remove()                  
                if settings_entrance['path_index'] == 'index': 
                    pd.DataFrame(z_bathy_B).plot(color='blue', linestyle='--', ax=ax) 
                    plt.axhline(y=np.nanpercentile(z_bathy_B, settings_entrance['sand_percentile']), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                    plt.ylabel(settings_entrance['path_index_id'] + ' Index [-]', fontsize=settings_entrance['axlabelsize'])
                    ax.get_legend().remove() 
                
                #histograms
                ax=plt.subplot(4,3,12)            
                if settings_entrance['path_index'] in ['swir', 'mndwi']:
                    seaborn.kdeplot(z_swir, shade=True,vertical=False, color='lightblue',bw=settings_entrance['hist_bw'], legend=False, lw=2, ax=ax)
                    plt.xlim(-0.1,1)
                    seaborn.kdeplot(z_swir_B, shade=True,vertical=False, color='orange',bw=settings_entrance['hist_bw'],legend=False, lw=2, ax=ax)
                    plt.axvline(x=np.nanpercentile(z_ndwi_B, settings_entrance['sand_percentile'] )    , color='orchid', linestyle='dotted', lw=2, alpha=1) 
                    plt.text(np.nanpercentile(z_ndwi_B, settings_entrance['sand_percentile'] )     , 0.5 ,  str(settings_entrance['sand_percentile']) + 'p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                                         
                    plt.ylabel('Probability density', fontsize=settings_entrance['axlabelsize'])
                    plt.xlabel('SWIR1 along A-D (lightblue) and C-D (orange) transect', fontsize=settings_entrance['axlabelsize']) 
                if settings_entrance['path_index'] in ['ndwi', 'nir']: 
                    seaborn.kdeplot(z_ndwi, shade=True,vertical=False, color='lightblue',bw=settings_entrance['hist_bw'], legend=False, lw=2, ax=ax)
                    plt.xlim(-1,0.5)
                    seaborn.kdeplot(z_ndwi_B, shade=True,vertical=False, color='orange',bw=settings_entrance['hist_bw'],legend=False, lw=2, ax=ax)               
                    plt.axvline(x=np.nanpercentile(z_ndwi_B, settings_entrance['sand_percentile'] )  , color='orchid', linestyle='dotted', lw=2, alpha=1) 
                    plt.text(np.nanpercentile(z_ndwi_B, settings_entrance['sand_percentile'] )   , 0.5 ,  str(settings_entrance['sand_percentile']) + 'p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                                         
                    plt.ylabel('Probability density', fontsize=settings_entrance['axlabelsize'])
                    plt.xlabel('NIR along A-D (lightblue) and C-D (orange) transect', fontsize=settings_entrance['axlabelsize'])     
                if settings_entrance['path_index'] == 'index': 
                    seaborn.kdeplot(z_bathy, shade=True,vertical=False, color='lightblue',bw=settings_entrance['hist_bw'], legend=False, lw=2, ax=ax)
                    #plt.xlim(-0.1,1)
                    seaborn.kdeplot(z_bathy_B, shade=True,vertical=False, color='orange',bw=settings_entrance['hist_bw'],legend=False, lw=2, ax=ax)
                    img_ndwi_perc = np.nanpercentile(z_bathy_B, settings_entrance['sand_percentile'] )                 
                    plt.axvline(x=img_ndwi_perc, color='orchid', linestyle='dotted', lw=2, alpha=1) 
                    plt.text(img_ndwi_perc , 0.5 ,  str(settings_entrance['sand_percentile']) + 'p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                                         
                    plt.ylabel('Probability density', fontsize=settings_entrance['axlabelsize'])
                    plt.xlabel(settings_entrance['path_index_id'] + ' index along A-D (lightblue) and C-D (orange) transect', fontsize=settings_entrance['axlabelsize'])                  

                fig.tight_layout() 
                fig.savefig(image_out_path + '/' + filenames[i][:-4]  + '_' + settings_entrance['path_index'] +'_based_' +
                            'ABexp_ ' + str(settings_entrance['AB_cost_raster_amp_exponent']) + '_XBexp_ ' + 
                            str(settings_entrance['XB_cost_raster_amp_exponent']) + '.png') 
                plt.close() 
  
    
            if settings_entrance['animation_plot_bool']:                          
                #setup the figure
                fig = plt.figure(figsize=(17,10)) 
                
                #plot RGB image
                ax=plt.subplot(2,2,1)
                im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)  
                plt.imshow(im_RGB, interpolation="bicubic") 
                plt.rcParams["axes.grid"] = False
                plt.title(satname + ' ' +str(dates_dict[filenames[i]].date()), fontsize=settings_entrance['axlabelsize'])
                ax.axis('off')
                plt.xlim(Xmin-settings_entrance['img_crop_adjsut'], Xmax+settings_entrance['img_crop_adjsut'])
                plt.ylim(Ymax + settings_entrance['img_crop_adjsut_Yax'],Ymin - settings_entrance['img_crop_adjsut_Yax']) 
                ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color=settings_entrance['transect_color'])
                ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color=settings_entrance['transect_color'])
                ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color=settings_entrance['transect_color'])
                plt.text(pts_pix_interp[0,0]+2, pts_pix_interp[0,1]+2,'A',horizontalalignment='left', color=settings_entrance['transect_color'] , fontsize=16)
                plt.text(pts_pix_interp[-1,0]+2, pts_pix_interp[-1,1]+2,'B',horizontalalignment='left', color=settings_entrance['transect_color'], fontsize=16)               
                ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color=settings_entrance['transect_color'])
                ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color=settings_entrance['transect_color'])
                ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color=settings_entrance['transect_color'])
                plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color=settings_entrance['transect_color'] , fontsize=16)
                plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color=settings_entrance['transect_color'], fontsize=16)                
                if len(Intersection_coords) >= 1:
                    ax.plot(Intersection_coords[0], Intersection_coords[1],"x", color='cyan', markersize=15, lw=1.5)
                    #plt.text(Intersection_coords[0]+2, Intersection_coords[1]+2,'X',horizontalalignment='left', color='cyan', fontsize=16)
                   
                ax=plt.subplot(2,2,2) 
                if settings_entrance['plot_tide_time_series'] & settings_entrance['tide_bool']:              
                    #ax.set_title('Tide level at img aquisition = ' + img_tide +  ' [m aMSL]', fontsize=settings_entrance['axlabelsize'])
                    ax.grid(which='major', linestyle=':', color='0.5')
                    ax.plot(tides_df_ss.index, tides_df_ss['tide_level'], '-', color='0.6')
                    ax.plot(sat_tides_df_ss.index, sat_tides_df_ss['tide_level'], '-o', color='k', ms=4, mfc='w',lw=1)
                    plt.axhline(y=sat_tides_df_ss['tide_level'][i], color='red', linestyle='dotted', lw=3, alpha=0.9) 
                    ax.plot(sat_tides_df_ss.index[i], sat_tides_df_ss['tide_level'][i], '-o', color='red', ms=12, mfc='w',lw=7)
                    ax.set_ylabel('tide level [m aMSL]', fontsize=settings_entrance['axlabelsize'])
                    ax.set_ylim(min(tides_df_ss['tide_level']), max(tides_df_ss['tide_level']))
                    
                elif settings_entrance['path_index'] == 'nir':  
                    plt.imshow(im_ms[:,:,3], cmap='seismic', vmin=0, vmax=0.8) 
                    plt.title('NIR', fontsize=settings_entrance['axlabelsize'])
                    ax.axis('off')
                    plt.xlim(Xmin-settings_entrance['img_crop_adjsut_Xax'], Xmax+settings_entrance['img_crop_adjsut_Xax'])
                    plt.ylim(Ymax + settings_entrance['img_crop_adjsut_Yax'],Ymin - settings_entrance['img_crop_adjsut_Yax']) 
                    ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
                    ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
                    ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
                    plt.text(pts_pix_interp[0,0]+2, pts_pix_interp[0,1]+2,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                    plt.text(pts_pix_interp[-1,0]+2, pts_pix_interp[-1,1]+2,'B',horizontalalignment='left', color='yellow', fontsize=16)               
                    ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
                    ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
                    ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
                    plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
                    plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)                
                    if len(Intersection_coords) >= 1:
                        ax.plot(Intersection_coords[0], Intersection_coords[1],"x", color='cyan', markersize=15, lw=1.5)        
                    if settings_entrance['plt_colorbars']:
                        plt.colorbar()
                
                elif settings_entrance['path_index'] == 'swir':  
                    plt.imshow(im_ms[:,:,3], cmap='seismic', vmin=0, vmax=0.8) 
                    plt.title('NIR', fontsize=settings_entrance['axlabelsize'])
                    ax.axis('off')
                    plt.xlim(Xmin-settings_entrance['img_crop_adjsut_Xax'], Xmax+settings_entrance['img_crop_adjsut_Xax'])
                    plt.ylim(Ymax + settings_entrance['img_crop_adjsut_Yax'],Ymin - settings_entrance['img_crop_adjsut_Yax']) 
                    ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
                    ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
                    ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
                    plt.text(pts_pix_interp[0,0]+2, pts_pix_interp[0,1]+2,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                    plt.text(pts_pix_interp[-1,0]+2, pts_pix_interp[-1,1]+2,'B',horizontalalignment='left', color='yellow', fontsize=16)               
                    ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
                    ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
                    ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
                    plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
                    plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)                
                    if len(Intersection_coords) >= 1:
                        ax.plot(Intersection_coords[0], Intersection_coords[1],"x", color='cyan', markersize=15, lw=1.5)        
                    if settings_entrance['plt_colorbars']:
                        plt.colorbar()
                        
                elif settings_entrance['path_index'] == 'ndwi':  
                    plt.imshow(im_ndwi, cmap='seismic', vmin=-1, vmax=1) 
                    plt.title('NDWI', fontsize=settings_entrance['axlabelsize'])
                    ax.axis('off')
                    plt.xlim(Xmin-settings_entrance['img_crop_adjsut_Xax'], Xmax+settings_entrance['img_crop_adjsut_Xax'])
                    plt.ylim(Ymax + settings_entrance['img_crop_adjsut_Yax'],Ymin - settings_entrance['img_crop_adjsut_Yax']) 
                    ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
                    ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
                    ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
                    plt.text(pts_pix_interp[0,0]+2, pts_pix_interp[0,1]+2,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                    plt.text(pts_pix_interp[-1,0]+2, pts_pix_interp[-1,1]+2,'B',horizontalalignment='left', color='yellow', fontsize=16)               
                    ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
                    ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
                    ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
                    plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
                    plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)                
                    if len(Intersection_coords) >= 1:
                        ax.plot(Intersection_coords[0], Intersection_coords[1],"x", color='cyan', markersize=15, lw=1.5)        
                    if settings_entrance['plt_colorbars']:
                        plt.colorbar()
                        
                ax=plt.subplot(2,2,3) 
                if settings_entrance['path_index'] in ['ndwi', 'nir']:  
                    pd.DataFrame(z_ndwi).plot(color='blue', linestyle='--', ax=ax) 
                    plt.axhline(y=np.nanpercentile(z_ndwi_B, 50), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    plt.ylabel('NDWI [-]', fontsize=settings_entrance['axlabelsize'])                    
                if settings_entrance['path_index'] in ['mndwi', 'swir', 'index']: 
                    pd.DataFrame(z_mndwi).plot(color='blue', linestyle='--', ax=ax)                
                    plt.axhline(y=np.nanpercentile(z_mndwi_B, settings_entrance['sand_percentile']) , xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)   
                    plt.ylabel('mNDWI [-]', fontsize=settings_entrance['axlabelsize'])                             
                    plt.ylim(-0.9,0.9) 
                if len(Intersection_coords) >= 1:               
                    plt.axvline(x=XB_distance_to_intersection , color='cyan', linestyle='--', lw=1.4, alpha=0.85) 
                    plt.text(XB_distance_to_intersection , -0.88 ,  'X', color='cyan', rotation=90, ha='right', va='bottom',fontsize=13, alpha=settings_entrance['vhline_transparancy'])
                plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=2, alpha=0.9) 
                #plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
                #plt.text(len(z_mndwi)-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
                plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                ax.get_legend().remove()
                
                ax=plt.subplot(2,2,4) 
                if settings_entrance['path_index'] in ['ndwi', 'nir']:  
                    pd.DataFrame(z_ndwi_B).plot(color='blue', linestyle='--', ax=ax) 
                    plt.axhline(y=np.nanpercentile(z_ndwi_B, 50), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    plt.ylabel('NDWI [-]', fontsize=settings_entrance['axlabelsize'])                  
                if settings_entrance['path_index'] in ['mndwi', 'swir', 'index']:  
                    pd.DataFrame(z_mndwi_B).plot(color='blue', linestyle='--', ax=ax)  
                    plt.axhline(y=np.nanpercentile(z_mndwi_B, 50), xmin=-1, xmax=1, color='orchid', linestyle='-.', lw=1.2, alpha=0.9)
                    plt.ylabel('mNDWI [-]', fontsize=settings_entrance['axlabelsize'])
                if len(Intersection_coords) >= 1:               
                    plt.axvline(x=AB_distance_to_intersection , color='cyan', linestyle='--', lw=1.4, alpha=0.85) 
                    plt.text(AB_distance_to_intersection , -0.88 ,  'X', color='cyan', rotation=90, ha='right', va='bottom',fontsize=13, alpha=settings_entrance['vhline_transparancy'])         
                plt.ylim(-0.9,0.9)
                plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=2, alpha=0.9)        
                #plt.text(1,0,'C',horizontalalignment='left', color='grey' , fontsize=labelsize)
                #plt.text(len(z_mndwi_B)-2,0,'D',horizontalalignment='right', color='grey' , fontsize=labelsize)
                plt.xlabel('Distance along transect [m]', fontsize=settings_entrance['axlabelsize'])
                ax.get_legend().remove()  
                
                fig.tight_layout() 
                
                anim_image_out_path = os.path.join(csv_out_path, 'auto_transects_simple')
                if not os.path.exists(anim_image_out_path):
                        os.makedirs(anim_image_out_path) 
            
                fig.savefig(anim_image_out_path + '/' + filenames[i][:-4]  + '_' + settings_entrance['path_index'] +'_based_' +
                            'ABexp_ ' + str(settings_entrance['AB_cost_raster_amp_exponent']) + '_XBexp_ ' + 
                            str(settings_entrance['XB_cost_raster_amp_exponent']) +  '.png') 
                plt.close() 
                
                    
    #gdf_all.crs = {'init':'epsg:'+str(image_epsg)} # looks like mistake. geoms as input to the dataframe should already have the output epsg. 
    gdf_all.crs = {'init': 'epsg:'+ str(settings['output_epsg'])}
    # convert from image_epsg to user-defined coordinate system
    #gdf_all = gdf_all.to_crs({'init': 'epsg:'+str(settings['output_epsg'])})
    # save as shapefile
    gdf_all.to_file(os.path.join(csv_out_path, sitename + '_entrance_lines_auto_' +
                                 settings_entrance['path_index'] +'_based.shp'), driver='ESRI Shapefile') 
    
    #save the data to csv            
    XS_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in XS.items() ])) 
    XS_df.to_csv(os.path.join(csv_out_path, sitename + '_XS' + 
                              '_entrance_lines_auto_' + settings_entrance['path_index'] +
                              '_based.csv')) 
    
    print('Entrance lines have been saved as a shapefile and in csv format for NDWI and mNDWI')


def bestThreshold(y_true,y_pred):
    thresh_df = pd.DataFrame(y_pred)
    thresh_df['clfd'] = 2
    best_thresh = None
    best_score = -1
    #for thresh in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08, 0.09]:
    for thresh in np.arange(0.01, 0.2,  0.01):
        thresh_df['clfd'][thresh_df.iloc[:, 0] >= thresh] = 1
        thresh_df['clfd'][thresh_df.iloc[:, 0] < thresh] = 0      
        score = f1_score(y_true, thresh_df['clfd'])
        if score > best_score:
            tn, fp, fn, tp = confusion_matrix(y_true, thresh_df['clfd']).ravel()
            Accuracy = (tp+tn)/(tp+fp+fn+tn)
            best_thresh = thresh
            best_score = score
    return best_score , Accuracy, tn, fp, fn, tp, best_thresh
 
    
 
def setup_classification_df(XS_df, Training_data_df,postprocess_params):   
    
    #Divide processed data into open and closed 
    XS_o_df = pd.DataFrame()
    #XS_o_gdf = pd.DataFrame()
    for date in Training_data_df[Training_data_df['Entrance_state'] == 'open'].index:
        XS_o_df = pd.concat([XS_o_df, pd.DataFrame(pd.DataFrame(XS_df.filter(regex=date)))], axis = 1)
        #satnameforloop = Training_data_df['satname'][Training_data_df.index == date] #this needed to be done since the date[:10] filter wasn't fully accurate (i.e. when say S2 and L8 were aquired on the same day, both get chosen)  
        #XS_o_df = pd.concat([XS_o_df, pd.DataFrame(pd.DataFrame(XS_df.filter(regex=date)).filter(regex='_' + satnameforloop[0] + '_'))], axis = 1)
        
        #XS_o_gdf = pd.concat([XS_o_gdf, XS_gdf[XS_gdf['date'] == date]], axis = 0)
    
    #write open across-berm transects out as shapefile for visualization
    #XS_o_gdf[XS_o_gdf['direction']=='XB'].to_file(csv_out_path  + '/open_AB_lines.shp', driver='ESRI Shapefile')     
    
    XS_c_df = pd.DataFrame()
    #XS_c_gdf = pd.DataFrame()
    for date in Training_data_df[Training_data_df['Entrance_state'] == 'closed'].index:
        XS_c_df = pd.concat([XS_c_df, pd.DataFrame(XS_df.filter(regex=date))], axis = 1)
        # satnameforloop = Training_data_df['satname'][Training_data_df.index == date] #this needed to be done since the date[:10] filter wasn't fully accurate (i.e. when say S2 and L8 were aquired on the same day, both get chosen)   
        # XS_c_df = pd.concat([XS_c_df, pd.DataFrame(XS_df.filter(regex=date).filter(regex='_' + satnameforloop[0] + '_'))], axis = 1)
                
        #XS_c_df =  XS_c_df.append(XS_df.filter(regex=date[:10]), ignore_index=True)
        #XS_c_df = pd.concat([XS_c_df, pd.DataFrame(XS_df.filter(regex=date[:10]))], axis = 1)
        #XS_c_gdf = pd.concat([XS_c_gdf, XS_gdf[XS_gdf['date'] == date]], axis = 0)
    
    
    directions = ['AB', 'XB']
    XBAB_XS_c_sums_df = pd.DataFrame()
    for direction in directions:
        #create a new dataframe that has a data timeseries index and the delta between the max of the XB transect and the ~50 percentile of the AB transect
        df2 = XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index'] + '_').filter(regex='_' + direction).dropna()
        df2b = XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna().max()  
        df4 = pd.DataFrame(XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_AB_').dropna().quantile(q=postprocess_params['metric_percentile'],
                              axis=0, numeric_only=True, interpolation='linear'))
              
        #keep only columns in df2 if they are also in df2b
        dfintm = XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna()
        newindex = []
        for index in df2.columns:
            newindex.append(index[:19])
        df2.columns = newindex 
        newindex = []
        for index in df4.index:
            newindex.append(index[:19])
        df4.index = newindex     
        newindex = []
        for index in dfintm.columns:
            newindex.append(index[:19])
        dfintm.columns = newindex 
        
        df2 = df2[dfintm.columns]
        df4 = df4.ix[dfintm.columns]
        
        newindex = []
        for index in df2b.index:
            newindex.append(index[:19])
        df2b.index = newindex    
        df2_dict = {}
        for XScolname in df2.columns:
            intersection = df2b[XScolname[:19]]       
            if not isinstance(intersection, float):
                intersection = intersection.iloc[0]
            if direction == 'AB':
                df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['AB_intersection_search_distance']),0) : np.int(intersection + postprocess_params['AB_intersection_search_distance'] )].min()         
            else: 
                df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['XB_intersection_search_distance']),0) : np.int(intersection + postprocess_params['XB_intersection_search_distance'] )].min()         
            
        df3 = pd.DataFrame.from_dict(df2_dict,  orient='index')        
        
        
        newindex = []
        for index in df4.index:
            newindex.append(pd.to_datetime(index[:19], format = '%Y-%m-%d-%H-%M-%S'))
        df3.index = newindex
        df4.index = newindex
        df5 = pd.concat([ df3,  df4], axis = 1, join='outer')
        df5.columns = ['maxindex','ABpercent']
        XS_c_sums_df = df5['ABpercent'] - df5['maxindex'] 
        
        
        XS_c_sums_df.columns = direction
        XBAB_XS_c_sums_df = pd.concat([ XBAB_XS_c_sums_df,  XS_c_sums_df ], axis = 1, join='outer')
    
    XBAB_XS_c_sums_df.columns = ['Along_berm_DTM', 'Across_berm_DTM']
    XBAB_XS_c_sums_df['user_entrance_state'] = 0
    
    
    directions = ['AB', 'XB']
    XBAB_XS_o_sums_df = pd.DataFrame()
    for direction in directions:
        #create a new dataframe that has a data timeseries index and the delta between the max of the XB transect and the ~50 percentile of the AB transect
        df2 = XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index'] + '_').filter(regex='_' + direction).dropna()
        df2b = XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna().max()  
        df4 = pd.DataFrame(XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_AB_').dropna().quantile(q=postprocess_params['metric_percentile'],
                              axis=0, numeric_only=True, interpolation='linear'))       
              
        #keep only columns in df2 if they are also in df2b
        dfintm = XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna()
        newindex = []
        for index in df2.columns:
            newindex.append(index[:19])
        df2.columns = newindex 
        newindex = []
        for index in df4.index:
            newindex.append(index[:19])
        df4.index = newindex     
        newindex = []
        for index in dfintm.columns:
            newindex.append(index[:19])
        dfintm.columns = newindex 
        
        df2 = df2[dfintm.columns]
        df4 = df4.ix[dfintm.columns]
        
        newindex = []
        for index in df2b.index:
            newindex.append(index[:19])
        df2b.index = newindex    
        df2_dict = {}
        for XScolname in df2.columns:
            intersection = df2b[XScolname[:19]]       
            if not isinstance(intersection, float):
                intersection = intersection.iloc[0]
            if direction == 'AB':
                df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['AB_intersection_search_distance']),0) : np.int(intersection + postprocess_params['AB_intersection_search_distance'] )].min()         
            else: 
                df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['XB_intersection_search_distance']),0) : np.int(intersection + postprocess_params['XB_intersection_search_distance'] )].min()         
            
            
        df3 = pd.DataFrame.from_dict(df2_dict,  orient='index')        
        
        newindex = []
        for index in df4.index:
            newindex.append(pd.to_datetime(index[:19], format = '%Y-%m-%d-%H-%M-%S'))
        df3.index = newindex
        df4.index = newindex
        df5 = pd.concat([ df3,  df4], axis = 1, join='outer')
        df5.columns = ['maxindex','ABpercent']
        XS_o_sums_df = df5['ABpercent'] - df5['maxindex'] 
         
        XS_o_sums_df.columns = direction
        XBAB_XS_o_sums_df = pd.concat([ XBAB_XS_o_sums_df,  XS_o_sums_df ], axis = 1, join='outer')
    
    XBAB_XS_o_sums_df.columns = ['Along_berm_DTM', 'Across_berm_DTM']
    XBAB_XS_o_sums_df['user_entrance_state'] = 1

    Classification_df = pd.DataFrame(pd.concat([XBAB_XS_o_sums_df, XBAB_XS_c_sums_df], axis = 0))
    Classification_df = Classification_df.dropna()
    
    return Classification_df
 

def classify_image_series_via_DTM(XS_df, direction, DTM_threshold, postprocess_params):
 
    #classify the full processed image series into open vs. closed binary entrance states
        
    #create a new dataframe that has a data timeseries index and the delta between the max of the XB transect and the ~50 percentile of the AB transect
    df2 = XS_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index'] + '_').filter(regex='_' + direction).dropna()
    df2b = XS_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna().max()  
    df4 = pd.DataFrame(XS_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_AB_').dropna().quantile(q=postprocess_params['metric_percentile'] ,
                          axis=0, numeric_only=True, interpolation='linear'))
          
    #keep only columns in df2 if they are also in df2b
    dfintm = XS_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna()
    newindex = []
    for index in df2.columns:
        newindex.append(index[:19])
    df2.columns = newindex 
    newindex = []
    for index in df4.index:
        newindex.append(index[:19])
    df4.index = newindex     
    newindex = []
    for index in dfintm.columns:
        newindex.append(index[:19])
    dfintm.columns = newindex 
    
    df2 = df2[dfintm.columns]
    df4 = df4.ix[dfintm.columns]
    
    newindex = []
    for index in df2b.index:
        newindex.append(index[:19])
    df2b.index = newindex    
    df2_dict = {}
    for XScolname in df2.columns:
        intersection = df2b[XScolname[:19]]       
        if not isinstance(intersection, float):
            intersection = intersection.iloc[0]
        if direction == 'AB':
            df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['AB_intersection_search_distance']),0) : np.int(intersection + postprocess_params['AB_intersection_search_distance'] )].min()         
        else: 
            df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['XB_intersection_search_distance']),0) : np.int(intersection + postprocess_params['XB_intersection_search_distance'] )].min()         
        
        
    df3 = pd.DataFrame.from_dict(df2_dict,  orient='index')        
     
    newindex = []
    for index in df4.index:
        newindex.append(pd.to_datetime(index[:19], format = '%Y-%m-%d-%H-%M-%S'))
    df3.index = newindex
    df4.index = newindex
    df5 = pd.concat([ df3,  df4], axis = 1, join='outer')
    df5.columns = ['maxindex','ABpercent']
    XS_sums_df = df5['ABpercent'] - df5['maxindex']
    
    XS_sums_df= pd.DataFrame(XS_sums_df)

    if direction == 'AB':
        XS_sums_df.columns = ['Along_berm_DTM']
        XS_sums_df['bin_entrance_state'] = 0
        XS_sums_df['bin_entrance_state'][XS_sums_df['Along_berm_DTM'] > DTM_threshold] = 1
    else:
        XS_sums_df.columns = ['Across_berm_DTM']      
        XS_sums_df['bin_entrance_state'] = 0
        XS_sums_df['bin_entrance_state'][XS_sums_df['Across_berm_DTM'] > DTM_threshold] = 1
    
    newindex = []
    for index in XS_sums_df.index:
        date = str(index)
        date = date.replace(" ", "-")
        date = date.replace(":", "-")
        newindex.append(date)
    XS_sums_df.index = newindex

    return XS_sums_df




def plot_entrancesat_results(XS_o_df, XS_c_df,XS_o_gdf, XS_c_gdf, settings, postprocess_params, metadata,  figure_out_path):

    #Fully automated part of the plot function    
    plt.style.use('classic')  
    
    filepath = SDS_tools.get_filepath(settings['inputs'],postprocess_params['satname_img'])
    filenames = metadata[postprocess_params['satname_img']]['filenames']
    epsg_dict = dict(zip(filenames, metadata[postprocess_params['satname_img']]['epsg']))      
            
    #plot font size and type
    ALPHA_figs = 1
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 12}
    matplotlib.rc('font', **font)
    
    params = {'labelsize': 'small',
             'axes.titlesize':'small',
             'labelsize' :'small',
             'ytick.labelsize':'small'}
    
    #plot settings
    plt.close('all') 
    fig = plt.figure(figsize=(11,11))  
    print('plotting Figure 1')
    
    ####################################
    #Plot the closed entrance states
    ####################################
    direction = 'XB'
    
    # row 1 pos 2   ################################################################################  
    Loopimage_c_date = XS_c_df.filter(regex=postprocess_params['satname_img']).filter(regex='_' + postprocess_params['spectral_index']).columns[0][:19]
    r = re.compile(".*" + Loopimage_c_date)    
    #combined_df.filter(regex=Gauge[3:]+'_')
    fn = SDS_tools.get_filenames(list(filter(r.match, filenames))[0],filepath, postprocess_params['satname_img'])
    im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, postprocess_params['satname_img'], settings['cloud_mask_issue'])
    
    # rescale image intensity for display purposes  
    im_plot = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)  
        
    image_epsg = epsg_dict[list(filter(r.match, filenames))[0]]
    shapes = load_shapes_as_ndarrays(settings['inputs']['location_shps']['layer'].values, settings['inputs']['location_shps'], postprocess_params['satname_img'],
                                     settings['inputs']['sitename'], settings['shapefile_EPSG'],
                                       georef, metadata, image_epsg)     
    
    x0, y0 = shapes['A'][1,:]
    x1, y1 = shapes['B'][1,:]
    Xmin,Xmax,Ymin,Ymax = get_bounding_box_minmax(shapes['A-B Mask'])
    
    direction =  'XB'
    XS_c_gdf_dir = XS_c_gdf[XS_c_gdf['direction']==direction]
            
    ax=plt.subplot(2,2,1)
    plt.title(postprocess_params['satname_img'] + ' ' + Loopimage_c_date + ' closed') 
    plt.imshow(im_plot, interpolation=postprocess_params['Interpolation_method']) 
    plt.rcParams["axes.grid"] = False
    plt.xlim(Xmin-postprocess_params['xaxisadjust'] , Xmax+postprocess_params['xaxisadjust'])
    plt.ylim(Ymax,Ymin) 
    #ax.grid(None)
    ax.axis('off')
    
    n=len(XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction ).columns)
    color=iter(cm.Oranges(np.linspace(0,1,n)))
    for i in range(0,n,1):    
        #plot the digitized entrance paths on top of images
        line = list(XS_c_gdf_dir[XS_c_gdf_dir['date'].str.contains(XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).columns[i][:19])].geometry.iloc[0].coords) ##!@check date functions
        df = pd.DataFrame(line)
        df = df.drop(columns=2)      
        pts_world_interp_reproj = SDS_tools.convert_epsg(df.values, settings['output_epsg'], image_epsg)
        df2 = pd.DataFrame(pts_world_interp_reproj)
        df2 = df2.drop(columns=2)  
        pts_pix_interp = SDS_tools.convert_world2pix(df2.values, georef)
        c= next(color)
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], linestyle = postprocess_params['linestyle'][0],lw=postprocess_params['linewidth'], color=c)
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko',   color=c)
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color=c)  
        if i==0:
            plt.text(pts_pix_interp[0,0]+3, pts_pix_interp[0,1]+3,'A',horizontalalignment='left', color=postprocess_params['closed_color'], fontsize=postprocess_params['labelsize'])
            plt.text(pts_pix_interp[-1,0]+3, pts_pix_interp[-1,1]+3,'B',horizontalalignment='left', color=postprocess_params['closed_color'], fontsize=postprocess_params['labelsize'])
    
    direction =  'AB'
    XS_c_gdf_dir = XS_c_gdf[XS_c_gdf['direction']==direction]
    
    ax=plt.subplot(2,2,2)
    plt.title(postprocess_params['satname_img'] + ' ' + Loopimage_c_date + ' closed') 
    plt.imshow(im_plot, interpolation=postprocess_params['Interpolation_method']) 
    plt.rcParams["axes.grid"] = False
    plt.xlim(Xmin-postprocess_params['xaxisadjust'] , Xmax+postprocess_params['xaxisadjust'])
    plt.ylim(Ymax,Ymin) 
    #ax.grid(None)
    ax.axis('off')
    
    n=len(XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).columns)
    color=iter(cm.Oranges(np.linspace(0,1,n)))
    for i in range(0,n,1):    
        #plot the digitized entrance paths on top of images - failing due to mixed up epsgs
        line = list(XS_c_gdf_dir[XS_c_gdf_dir['date'].str.contains(XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).columns[i][:19])].geometry.iloc[0].coords)
        df = pd.DataFrame(line)
        df = df.drop(columns=2)      
        pts_world_interp_reproj = SDS_tools.convert_epsg(df.values, settings['output_epsg'], image_epsg)
        df2 = pd.DataFrame(pts_world_interp_reproj)
        df2 = df2.drop(columns=2)  
        pts_pix_interp = SDS_tools.convert_world2pix(df2.values, georef)
        c= next(color)
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], linestyle=postprocess_params['linestyle'][0],lw=0.5, color=c)
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko',   color=c)
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color=c)  
        if i==0:
            plt.text(pts_pix_interp[0,0]+3, pts_pix_interp[0,1]+3,'C',horizontalalignment='left', color=postprocess_params['closed_color'] , fontsize=postprocess_params['labelsize'])
            plt.text(pts_pix_interp[-1,0]+3, pts_pix_interp[-1,1]+3,'D',horizontalalignment='left', color=postprocess_params['closed_color'], fontsize=postprocess_params['labelsize'])
    
    ####################################
    #Plot the open entrance states
    #################################### 
            
    # row 1 pos 2   ################################################################################         
    Loopimage_c_date = XS_o_df.filter(regex=postprocess_params['satname_img']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).columns[0][:19]
    r = re.compile(".*" + Loopimage_c_date)    
    #combined_df.filter(regex=Gauge[3:]+'_')
    fn = SDS_tools.get_filenames(list(filter(r.match, filenames))[0],filepath, postprocess_params['satname_img'])
    #print(fn)
    im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, postprocess_params['satname_img'], settings['cloud_mask_issue'])
    # rescale image intensity for display purposes
    
    im_plot = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)  
        
    image_epsg = epsg_dict[list(filter(r.match, filenames))[0]]
    shapes = load_shapes_as_ndarrays(settings['inputs']['location_shps']['layer'].values, settings['inputs']['location_shps'], postprocess_params['satname_img'],  settings['inputs']['sitename'], settings['shapefile_EPSG'],
                                       georef, metadata, image_epsg)  

       
    x0, y0 = shapes['A'][1,:]
    x1, y1 = shapes['B'][1,:]
    Xmin,Xmax,Ymin,Ymax = get_bounding_box_minmax(shapes['A-B Mask'])
    
    direction =  'XB'
    XS_o_gdf_dir = XS_o_gdf[XS_o_gdf['direction']==direction]
    
    ax=plt.subplot(2,2,3)
    plt.title(postprocess_params['satname_img'] + ' ' + Loopimage_c_date + ' open') 
    plt.imshow(im_plot, interpolation=postprocess_params['Interpolation_method']) 
    plt.rcParams["axes.grid"] = False
    plt.xlim(Xmin-postprocess_params['xaxisadjust'] , Xmax+postprocess_params['xaxisadjust'])
    plt.ylim(Ymax,Ymin) 
    #ax.grid(None)
    ax.axis('off')
    
    n=len(XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).columns)
    color=iter(cm.Blues(np.linspace(0,1,n)))
    for i in range(0,n,1):
        #plot the digitized entrance paths on top of images - failing due to mixed up epsgs
        line = list(XS_o_gdf_dir[XS_o_gdf_dir['date'].str.contains(XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).columns[i][:19])].geometry.iloc[0].coords)
        df = pd.DataFrame(line)
        df = df.drop(columns=2)      
        pts_world_interp_reproj = SDS_tools.convert_epsg(df.values, settings['output_epsg'], image_epsg)
        df2 = pd.DataFrame(pts_world_interp_reproj)
        df2 = df2.drop(columns=2)  
        pts_pix_interp = SDS_tools.convert_world2pix(df2.values, georef)
        c= next(color)
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], linestyle=postprocess_params['linestyle'][0],lw=0.5, color=c)
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko',   color=c)
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color=c)  
        if i==0:
            plt.text(pts_pix_interp[0,0]+3, pts_pix_interp[0,1]+3,'A',horizontalalignment='left', color=postprocess_params['open_color'] , fontsize=postprocess_params['labelsize'])
            plt.text(pts_pix_interp[-1,0]+3, pts_pix_interp[-1,1]+3,'B',horizontalalignment='left', color=postprocess_params['open_color'], fontsize=postprocess_params['labelsize'])
    
    direction =  'AB'
    XS_o_gdf_dir = XS_o_gdf[XS_o_gdf['direction']==direction]
    ax=plt.subplot(2,2,4)
    plt.title(postprocess_params['satname_img'] + ' ' + Loopimage_c_date + ' open') 
    plt.imshow(im_plot, interpolation=postprocess_params['Interpolation_method']) 
    plt.rcParams["axes.grid"] = False
    plt.xlim(Xmin-postprocess_params['xaxisadjust'] , Xmax+postprocess_params['xaxisadjust'])
    plt.ylim(Ymax,Ymin) 
    #ax.grid(None)
    ax.axis('off')
     
    n=len(XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).columns)
    color=iter(cm.Blues(np.linspace(0,1,n)))
    for i in range(0,n,1):
        #plot the digitized entrance paths on top of images - failing due to mixed up epsgs
        line = list(XS_o_gdf_dir[XS_o_gdf_dir['date'].str.contains(XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).columns[i][:19])].geometry.iloc[0].coords)
        df = pd.DataFrame(line)
        df = df.drop(columns=2)      
        pts_world_interp_reproj = SDS_tools.convert_epsg(df.values, settings['output_epsg'], image_epsg)
        df2 = pd.DataFrame(pts_world_interp_reproj)
        df2 = df2.drop(columns=2)  
        pts_pix_interp = SDS_tools.convert_world2pix(df2.values, georef)
        c= next(color)
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], linestyle=postprocess_params['linestyle'][0],lw=0.5, color=c)
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko',   color=c)
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color=c)  
        if i==0:
            plt.text(pts_pix_interp[0,0]+3, pts_pix_interp[0,1]+3,'C',horizontalalignment='left', color=postprocess_params['open_color'] , fontsize=postprocess_params['labelsize'])
            plt.text(pts_pix_interp[-1,0]+3, pts_pix_interp[-1,1]+3,'D',horizontalalignment='left', color=postprocess_params['open_color'], fontsize=postprocess_params['labelsize'])       
    
    fig.tight_layout()   
    fig.savefig(os.path.join(figure_out_path,'Figure_1_'  + postprocess_params['spectral_index'] + '_' + datetime.now().strftime("%d-%m-%Y") + '_' + postprocess_params['satnames_string'] + '.png'), dpi=400)          
    plt.close()
    
    
    
    
    
    #plot the mNDWI transects & delta-to-median parameter ################################################################################      
    
    fig = plt.figure(figsize=(12,14))  
    print('plotting Figure 2')
    
    direction =  'AB'  
    ax=plt.subplot(4,1,1)      
    XS_c_df.filter(regex=postprocess_params['satnames_XS']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).dropna().plot(color=postprocess_params['closed_color'], linestyle = postprocess_params['linestyle'][0],lw=1.5,alpha=0.5, ax=ax)
    XS_o_df.filter(regex=postprocess_params['satnames_XS']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).dropna().plot(color=postprocess_params['open_color'],  linestyle = postprocess_params['linestyle'][0],lw=1.5,alpha=0.5,ax=ax)
    
    plt.ylim(-0.9,0.9)
    #plt.title('postprocess_params['spectral_index'] + ' along transects') 
    #plt.legend()
    plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=0.5) 
    plt.text(1,0,'C',horizontalalignment='left', color='grey' , fontsize=postprocess_params['labelsize'])
    plt.xlabel('Distance along along-berm transects [m]')
    plt.ylabel(postprocess_params['spectral_index'])
    ax.get_legend().remove()  
    
    
    # row 2 pos 2  ################################################################################   
    #plot the mNDWI sums along the transects over time
    ax=plt.subplot(4,1,3)  
    
    
    #create a new dataframe that has a data timeseries index and the delta between the max of the XB transect and the ~50 percentile of the AB transect
    df2 = XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index'] + '_').filter(regex='_' + direction).dropna()
    df2b = XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna().max()  
    df4 = pd.DataFrame(XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_AB_').dropna().quantile(q=postprocess_params['metric_percentile'] ,
                          axis=0, numeric_only=True, interpolation='linear'))
          
    #keep only columns in df2 if they are also in df2b
    dfintm = XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna()
    newindex = []
    for index in df2.columns:
        newindex.append(index[:19])
    df2.columns = newindex 
    newindex = []
    for index in df4.index:
        newindex.append(index[:19])
    df4.index = newindex     
    newindex = []
    for index in dfintm.columns:
        newindex.append(index[:19])
    dfintm.columns = newindex 
    
    df2 = df2[dfintm.columns]
    df4 = df4.ix[dfintm.columns]
    
    newindex = []
    for index in df2b.index:
        newindex.append(index[:19])
    df2b.index = newindex    
    df2_dict = {}
    for XScolname in df2.columns:
        intersection = df2b[XScolname[:19]]       
        if not isinstance(intersection, float):
            intersection = intersection.iloc[0]
        df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['AB_intersection_search_distance']),0) : np.int(intersection +  postprocess_params['AB_intersection_search_distance'])].min()         
    df3 = pd.DataFrame.from_dict(df2_dict,  orient='index')        
    
    newindex = []
    for index in df4.index:
        newindex.append(pd.to_datetime(index[:19], format = '%Y-%m-%d-%H-%M-%S'))
    df3.index = newindex
    df4.index = newindex
    df5 = pd.concat([ df3,  df4], axis = 1, join='outer')
    df5.columns = ['maxindex','ABpercent']
    XS_c_sums_df = df5['ABpercent'] - df5['maxindex'] 
        
      
    newindex = []
    for index in XS_c_sums_df.index:
        newindex.append(pd.to_datetime(index, format = '%Y-%m-%d-%H-%M-%S'))
    XS_c_sums_df.index = newindex
    
    XS_c_sums_df = pd.DataFrame(XS_c_sums_df)
    
    
    
    #create a new dataframe that has a data timeseries index and the delta between the max of the XB transect and the ~50 percentile of the AB transect
    df2 = XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index'] + '_').filter(regex='_' + direction).dropna()
    df2b = XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna().max()  
    df4 = pd.DataFrame(XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_AB_').dropna().quantile(q=postprocess_params['metric_percentile'] ,
                          axis=0, numeric_only=True, interpolation='linear'))   
         
    #keep only columns in df2 if they are also in df2b
    dfintm = XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna()
    newindex = []
    for index in df2.columns:
        newindex.append(index[:19])
    df2.columns = newindex 
    newindex = []
    for index in df4.index:
        newindex.append(index[:19])
    df4.index = newindex     
    newindex = []
    for index in dfintm.columns:
        newindex.append(index[:19])
    dfintm.columns = newindex 
    
    df2 = df2[dfintm.columns]
    df4 = df4.ix[dfintm.columns]
    
    newindex = []
    for index in df2b.index:
        newindex.append(index[:19])
    df2b.index = newindex 
    df2_dict = {}
    for XScolname in df2.columns:
        intersection = df2b[XScolname[:19]]       
        if not isinstance(intersection, float):
            intersection = intersection.iloc[0]
        df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['AB_intersection_search_distance']),0) : np.int(intersection + postprocess_params['AB_intersection_search_distance'])].min()  
    df3 = pd.DataFrame.from_dict(df2_dict,  orient='index')  
    
    newindex = []
    for index in df4.index:
        newindex.append(pd.to_datetime(index[:19], format = '%Y-%m-%d-%H-%M-%S'))
    df3.index = newindex
    df4.index = newindex
    df5 = pd.concat([ df3,  df4], axis = 1, join='outer')
    df5.columns = ['maxindex','ABpercent']
    XS_o_sums_df = df5['ABpercent'] - df5['maxindex']   
    
    newindex = []
    for index in XS_o_sums_df.index:
        newindex.append(pd.to_datetime(index, format = '%Y-%m-%d-%H-%M-%S'))
        
    XS_o_sums_df.index = newindex
    XS_o_sums_df = pd.DataFrame(XS_o_sums_df)
    
    XS_c_sums_AB_df = XS_c_sums_df
    XS_o_sums_AB_df = XS_o_sums_df
    
    XS_co_sums_AB_df = XS_c_sums_AB_df.append(XS_o_sums_AB_df )
    XS_co_sums_AB_df.plot(color='grey', style='--',lw=1, alpha=0.5, ax=ax)
    
    XS_c_sums_df.plot(color=postprocess_params['closed_color'],style='.',lw=postprocess_params['markersize'],  ax=ax)
    XS_o_sums_df.plot(color=postprocess_params['open_color'], style='.',lw=postprocess_params['markersize'], ax=ax)
        
    plt.ylim(XS_c_sums_df.min().min(),XS_o_sums_df.max().max())
    plt.ylabel('Along-berm Delta-to-median')
    ax.get_legend().remove()  
    
    
    # row 3 pos 1  ################################################################################                   
    #plot the mNDWI transects
    direction =  'XB'     
    ax=plt.subplot(4,1,2)
    XS_c_df.filter(regex=postprocess_params['satnames_XS']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).dropna().plot(color=postprocess_params['closed_color'], linestyle = postprocess_params['linestyle'][0],lw=1.5,alpha=0.5, ax=ax)
    XS_o_df.filter(regex=postprocess_params['satnames_XS']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_' + direction).dropna().plot(color=postprocess_params['open_color'],  linestyle= postprocess_params['linestyle'][0],lw=1.5,alpha=0.5,ax=ax)
    
    plt.ylim(-0.9,0.9)
    plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=0.5) 
    plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=postprocess_params['labelsize'])
    plt.xlabel('Distance along across-berm transects [m]')
    plt.ylabel(postprocess_params['spectral_index'])
    ax.get_legend().remove()  
    
    
    # row 2 pos 2  ################################################################################   
    ax=plt.subplot(4,1,4)
    #XB_intersection_search_distance = 300
    #plot the mNDWI sums along the transects over time
        
    #create a new dataframe that has a data timeseries index and the delta between the max of the XB transect and the ~50 percentile of the AB transect
    df2 = XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index'] + '_').filter(regex='_' + direction).dropna()
    df2b = XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna().max()  
    df4 = pd.DataFrame(XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_AB_').dropna().quantile(q=postprocess_params['metric_percentile'] ,
                          axis=0, numeric_only=True, interpolation='linear'))
          
    #keep only columns in df2 if they are also in df2b
    dfintm = XS_c_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna()
    newindex = []
    for index in df2.columns:
        newindex.append(index[:19])
    df2.columns = newindex 
    newindex = []
    for index in df4.index:
        newindex.append(index[:19])
    df4.index = newindex     
    newindex = []
    for index in dfintm.columns:
        newindex.append(index[:19])
    dfintm.columns = newindex 
    
    df2 = df2[dfintm.columns]
    df4 = df4.ix[dfintm.columns]
    
    newindex = []
    for index in df2b.index:
        newindex.append(index[:19])
    df2b.index = newindex    
    df2_dict = {}
    for XScolname in df2.columns:
        #XScolname = '2014-08-11'
        intersection = df2b[XScolname[:19]]       
        if not isinstance(intersection, float):
            intersection = intersection.iloc[0]
        df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['XB_intersection_search_distance']),0): np.int(intersection + postprocess_params['XB_intersection_search_distance'])].max()          
    df3 = pd.DataFrame.from_dict(df2_dict,  orient='index')        
    
    newindex = []
    for index in df4.index:
        newindex.append(pd.to_datetime(index[:19], format = '%Y-%m-%d-%H-%M-%S'))
    df3.index = newindex
    df4.index = newindex
    df5 = pd.concat([ df3,  df4], axis = 1, join='outer')
    df5.columns = ['maxindex','ABpercent']
    XS_c_sums_df = df5['ABpercent'] - df5['maxindex']  
    
    
    #create a new dataframe that has a data timeseries index and the delta between the max of the XB transect and the ~50 percentile of the AB transect
    df2 = XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index'] + '_').filter(regex='_' + direction).dropna()
    df2b = XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna().max()  
    df4 = pd.DataFrame(XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_' + postprocess_params['spectral_index']).filter(regex='_AB_').dropna().quantile(q= postprocess_params['metric_percentile'],
                          axis=0, numeric_only=True, interpolation='linear'))
          
    #keep only columns in df2 if they are also in df2b
    dfintm = XS_o_df.filter(regex=postprocess_params['satnames']).filter(regex='_distance_to_intersection').filter(regex='_' + direction).dropna()
    newindex = []
    for index in df2.columns:
        newindex.append(index[:19])
    df2.columns = newindex 
    newindex = []
    for index in df4.index:
        newindex.append(index[:19])
    df4.index = newindex     
    newindex = []
    for index in dfintm.columns:
        newindex.append(index[:19])
    dfintm.columns = newindex 
    
    df2 = df2[dfintm.columns]
    df4 = df4.ix[dfintm.columns]
    
    newindex = []
    for index in df2b.index:
        newindex.append(index[:19])
    df2b.index = newindex 
    df2_dict = {}
    for XScolname in df2.columns:
        #XScolname = '2014-08-11'
        intersection = df2b[XScolname[:19]]       
        if not isinstance(intersection, float):
            intersection = intersection.iloc[0]
        df2_dict[XScolname] = df2[XScolname][max(np.int(intersection - postprocess_params['XB_intersection_search_distance']),0) : np.int(intersection + postprocess_params['XB_intersection_search_distance'])].max()  
    df3 = pd.DataFrame.from_dict(df2_dict,  orient='index')  
    
    newindex = []
    for index in df4.index:
        newindex.append(pd.to_datetime(index[:19], format = '%Y-%m-%d-%H-%M-%S'))
    df3.index = newindex
    df4.index = newindex
    df5 = pd.concat([ df3,  df4], axis = 1, join='outer')
    df5.columns = ['maxindex','ABpercent']
    XS_o_sums_df = df5['ABpercent'] - df5['maxindex']  
                
    XS_c_sums_XB_df = XS_c_sums_df
    XS_o_sums_XB_df = XS_o_sums_df
    
    XS_co_sums_XB_df = XS_c_sums_XB_df.append(XS_o_sums_XB_df)
    XS_co_sums_XB_df.plot(color='grey', style='--',lw=1.5, alpha=0.5, ax=ax)
    
    XS_c_sums_df.plot(color=postprocess_params['closed_color'],style='.',lw=postprocess_params['markersize'], alpha = 1.5, ax=ax)
    XS_o_sums_df.plot(color=postprocess_params['open_color'], style='.',lw=postprocess_params['markersize'], alpha = 1.5, ax=ax)
    
    plt.ylabel('Across-berm Delta-to-median')
         
    fig.tight_layout()   
    fig.savefig(os.path.join(figure_out_path,'Figure_2_'  + postprocess_params['spectral_index'] + '_' + datetime.now().strftime("%d-%m-%Y") + '_' + postprocess_params['satnames_string'] + '.png'), dpi=400)          
    plt.close()
    
    
    #combine the processed data into a single dataframe and export as a csv file
    XS_c_sums_XB_df1 = pd.DataFrame(XS_c_sums_XB_df)
    XS_o_sums_XB_df1 = pd.DataFrame(XS_o_sums_XB_df)   
    
    XS_c_sums_XB_df1['bin_entrance_state'] = 0
    XS_o_sums_XB_df1['bin_entrance_state'] = 1
    XS_c_sums_XB_df1['entrance_state'] = 'closed'
    XS_o_sums_XB_df1['entrance_state'] = 'open'
    XS_co_sums_XB_cfd_df = XS_c_sums_XB_df1.append(XS_o_sums_XB_df1)
    XS_co_sums_XB_cfd_df.columns = ['Across_berm','bin_entrance_state', 'entrance_state']
    
    XS_c_sums_AB_df1 = pd.DataFrame(XS_c_sums_AB_df)
    XS_o_sums_AB_df1 = pd.DataFrame(XS_o_sums_AB_df)     
    XS_co_sums_AB_cfd_df = XS_c_sums_AB_df1.append(XS_o_sums_AB_df1)
    XS_co_sums_AB_cfd_df.columns = ['Along_berm']
    
    XS_co_sums_AB_cfd_df = XS_co_sums_AB_cfd_df.sort_index(axis=0)
    XS_co_sums_XB_cfd_df = XS_co_sums_XB_cfd_df.sort_index(axis=0)
    XS_sums_ABXB_df = pd.DataFrame(pd.concat([XS_co_sums_AB_cfd_df, XS_co_sums_XB_cfd_df], axis = 1))
    
    #export full dataframe with all the transects and delta to percentiles
    XS_sums_ABXB_df.to_csv(os.path.join(figure_out_path, settings['inputs']['sitename'] + '_postprocessed_summary_df_' + postprocess_params['spectral_index'] + '.csv'))
              
        
        

    
###################################################################################################
# Tide functions
###################################################################################################

def compute_tide(coords,date_range,time_step,ocean_tide,load_tide):
    'compute time-series of water level for a location and dates using a time_step'
    # list of datetimes (every timestep)
    dates = []
    date = date_range[0]
    while date <= date_range[1]:
        dates.append(date)
        date = date + timedelta(seconds=time_step)
    # convert list of datetimes to numpy dates
    dates_np = np.empty((len(dates),), dtype='datetime64[us]')
    for i,date in enumerate(dates):
        dates_np[i] = datetime(date.year,date.month,date.day,date.hour,date.minute,date.second)
    lons = coords[0]*np.ones(len(dates))
    lats = coords[1]*np.ones(len(dates))
    # compute heights for ocean tide and loadings
    ocean_short, ocean_long, min_points = ocean_tide.calculate(lons, lats, dates_np)
    load_short, load_long, min_points = load_tide.calculate(lons, lats, dates_np)
    # sum up all components and convert from cm to m
    tide_level = (ocean_short + ocean_long + load_short + load_long)/100

    return dates, tide_level


def compute_tide_dates(coords,dates,ocean_tide,load_tide):
    'compute time-series of water level for a location and dates (using a dates vector)'
    dates_np = np.empty((len(dates),), dtype='datetime64[us]')
    for i,date in enumerate(dates):
        dates_np[i] = datetime(date.year,date.month,date.day,date.hour,date.minute,date.second)
    lons = coords[0]*np.ones(len(dates))
    lats = coords[1]*np.ones(len(dates))
    # compute heights for ocean tide and loadings
    ocean_short, ocean_long, min_points = ocean_tide.calculate(lons, lats, dates_np)
    load_short, load_long, min_points = load_tide.calculate(lons, lats, dates_np)
    # sum up all components and convert from cm to m
    tide_level = (ocean_short + ocean_long + load_short + load_long)/100

    return tide_level