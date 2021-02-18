#==========================================================#
# Reconstructing historic entrance states of intermittent estuaries 
#==========================================================#

# Valentin Heimhuber, Water Research Laboratory, University of New South Wales, 2020

#%% Step 1: Initial algorithm settings & imagery download

#load modules
import os
from entrancesat import  SDS_entrance 
from coastsat import SDS_download
import pandas as pd
import glob
import pickle

# filepath where data will be stored
filepath_data = os.path.join(os.getcwd(), 'data') #user input required | change this path to the location where you want to store the data (can be outside of ../Entrencesat)

#sitename as specified in the input input_locations.shp
sitename = 'DURRAS' #user input required
site_shapefile_name = 'input_locations.shp' #user input required | change this if a new shapefile was created with the site configurations

#this parameter is used to distinguish progressive 'sets' of analysis that may be based on different seed and receiver point configurations
#note that within this set of results, a unique directory is created for each path finding index
Analysis_version = 'V1'   #user input required

# date range for analysis
dates = ['1985-01-01', '2020-11-01']   #user input required

# satellite missions
sat_list = ['L5','L7','L8','S2'] #user input required

satnames = '|'.join(sat_list)
satnames_string = '_'.join(sat_list)


#load shapefile that contains specific shapes for each ICOLL site as per readme file
Site_shps, layers, BBX_coords = SDS_entrance.load_shapes(site_shapefile_name, sitename)

      
# put all the inputs into a dictionnary
inputs = {
    'polygon': BBX_coords,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename,
    'filepath': filepath_data,
    'location_shps': Site_shps,
    'analysis_vrs' : Analysis_version
        }

# retrieve satellite images from GEE
#metadata = SDS_download.retrieve_images(inputs) #user input required (hash this line if you have already downloaded the data)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs) 
    
# general settings
settings = { 
    # general parameters:
    'cloud_thresh': 0.05,        # threshold on maximum cloud cover
    'output_epsg': 3577,       # epsg code of spatial reference system desired for the spatial output files
    'shapefile_EPSG' : 4326,     #epsg of shapefile containing sites and path finding seed and receiver points
    'use_fes_data': False,      # if the FES model was installed sucessfully, choose whether to include it in the analysis (True) or not (False). 
    'filepath_fes' : r"H:\Downloads\fes-2.9.1-Source\data\fes2014",
    # add the inputs defined previously
    'inputs': inputs,
    #advanced
    'cloud_mask_issue': True,  # switch this parameter to True if sand pixels are masked (in black) on many images 
    }      

    
    
    
    
    
#%%  Step 2: Create training data
"""
#create training data
In this step, a training data set is created via visual inspection of images. 
It is recommended to:
    -generate a training dataset of at least 10 open and 10 closed entrances. More training data will typically lead to more accurate classification results. 
    -Keep the number of open and closed images roughtly equal (this can be done using 'skip')
    -create at least two equally sized training datasets. One for the Landsat group (including 5,7 and 8) and one for S2. 
     Satellites can be skipped via Esc. 
"""
settings_training =  { # set parameters for training data generation
                    'shuffle_training_imgs':True,   # if True, images during manual/visual detection of entrance states are shuffled (in time) to provide a more independent sample
                    'save_figure': True,        # if True, saves a figure for each trained image     
                    'username' : 'EntranceSat_VH2', # in case multiple analysts create training data or one analyst creating multiple training datasets, this can be used as a distinguishing variable.
                      }

# only rerun this step if you have not already generated a set of training data (i.e., only run once)
Training_data_df = SDS_entrance.create_training_data(metadata, settings, settings_training)

 





#%%  Step 3: generate tide time series for site (if use_fes_data = True in settings)

#load tide data for analysis period and at all image acquisition times from the FES2014 global tide model
if settings['use_fes_data']:
    sat_tides_df, tides_df = SDS_entrance.load_FES_tide(settings, sat_list, metadata) 
else:
    tides_df = pd.DataFrame()       
    sat_tides_df = pd.DataFrame()






#%%  Step 4: find transects automatically and write results to dataframe and pickle files
"""
This is the major automated processing step of the algorithm consisting of: 
    -automated image pre-processing
    -along-berm and across-berm path finding and export of paths via ESRI shapefile
    -extraction of NIR, SWIR1, NDWI and mNDWI along each transect and export via csv and pkl files
    -export of a result dashboard .png for each image showing all important entrance detection features 
    -the results are stored in an automatically generated directory
    -for each algorithm configuration, the SDS_entrance.automated_entrance_paths function only has to be executed once
    -durin postprocessing, the results are then read in based on the parameters provided in settings_entrance
    -during parameter tuning, it is recommended to use the ''number_of_images parameter to limit the number of images being processed here
"""

# set parameters for automated entrance detection 
settings_entrance =  {   
                  
    #key algorithm parameters
    'path_index': 'swir',                   #band/index used for pathfinding
    'sand_percentile': 50 ,                #percentile of sand to plot in addition to 10th, 20th and 30th (which are always done)
    'XB_cost_raster_amp_exponent': 25,     #The cost raster based on 'path_index' will be exponentiated with this factor before path finding across berm
    'AB_cost_raster_amp_exponent': 25,     #The cost raster based on 'path_index' will be exponentiated with this factor before path finding along berm
    'cloud_cover_ROIonly' : True ,         #discard images based on cloud cover over the entrance area only instead of cloud cover over whole image/lagoon
    'use_berm_mask_for_AB' : True,         #use a separate mask for along berm path finding - recommended if there is vegetation around the entrance
    'number_of_images':2000,               #nr of images to process - if it exceeds len(images) all images will be processed. Applied to each satellite so 5 -> 20 images processed
    'use_straight_lines': False,             # insead of path finding, simply use a straightline, which might be useful for application not concerned with coastal entrances   
            
    #processing troubleshooting
    #sometimes specific images may cause the code to crash. If that image nr is included here it will be skipped when you rerun the algorithm 
    'skip_img_L8': [0],                   
    'skip_img_L7': [0],
    'skip_img_L5': [0],
    'skip_img_S2': [0], 
    
    #extract a different spectral index along the transects in addition to NDWI and mNDWI: 
    'index_id': 'bandratio',              #options are NIR, 'bandratio'|'ImprovedNIRwithSAC'|'NIRoverRedwithSAC'|'greenminusred'
    'band1': 1,                           #if index_id = bandratio, band1 is divided by band2
    'band2': 0,                           #bands 0,1,2,3,4 are blue, green, red, NIR, SWIR1
    
    #plotting/styling parameters - can be left as is
    'plot_bool': True ,                   #create the output plots in addition to csv files? 
    'transect_color' :  'black' ,         #transect color in plots | yellow or black are recommended
    'plt_colorbars': False,               #plt colorbars for index raster plots? Typically best to avoid to maximize the image plot area
    'plot_tide_histogram' : False,        # plot histograms of the tide time series and tide levels at satellite image dates
    'plot_entrance_bbx': True ,           #plot the entrance bounding box ontop of the RGB image for reference
    'img_crop_adjsut': 0,                 #nr of pixels to add on each side (along x axis) to the cropped image to fill out plt space. needs to be adjusted for each site                       
    'vhline_transparancy': 0.8 ,          #transparancy of v and h lines in the output plots
    'hist_bw': 0.05,                      #parameter for histogram smoothing in the output plots
    'fontsize' : 25 ,      #10            #size of fonts in plot
    'labelsize' :40   ,       #26          #size of text lables
    'axlabelsize': 20 ,                   #sie of axis labels    
    
    #plotting parameters for an additional, simpler output plot that is good for creating animations or illisrations
    'animation_plot_bool': True,          #output a second set of plots with a simpler plot layout and fewer windows useful for animations
    'plot_tide_time_series': True,        #plot a time series of the tide + tide during image instead of spectral index in top right window of animation plots
    'img_crop_adjsut_Xax': 0,            #X and Y axis adjustment factors to fit the cropped images better on the multipanel plots. These require experimentation to set properly. 
    'img_crop_adjsut_Yax': 10
    }

# run this function only if the current path finding settings haven't been processed yet (i.e., run only once)
#SDS_entrance.automated_entrance_paths(metadata, settings, settings_entrance, tides_df , sat_tides_df)








#%%  Step 5: Post-processing & analysis
"""
Post-processing and data analysis part of the toolkit. 
    This involves the calculation of the delta-to-median (DTM) parameter from the automatically traced along-berm and across-berm transects of Step 4
    Based on the user generated training dataset, an optimal classification threshold is identified for the DTM series (5.2) 
    This threshold is then used to classify the full image series into open vs. closed entrane states. (5.2)
    The automated entrance state detections can be manually checked in step 5.3, which is optional. 
    The resulting time series are plotted in a variety of different plots for analysis (5.4)
    They are also written out in the form of csv files which can be used as the basis for additional user analyses (5.4)
"""   
  
########################## 
#5.1 Set parameters for processing and load all required datasets
########################## 
  
#set processing parameters
postprocess_params = { 
    #processing parameters - consider carefully 
    'Postprocessing_version' : 'V1',           #this is a unique identifier used to distinguish different postprocessing versions (e.g. based on different spectral index)
    'spectral_index' : 'mndwi',                 #band/index used for inferring entrance states via delta-to-median parameter: options are ndwi, nir, swir or mndwi
    'metric_percentile' : 0.5 ,                 #which percentile to use for 'delta to median' parameter. Typically recommended to be 0.5, which is the median.  
    'AB_intersection_search_distance' : 100,    #window on either side of the AB intersection to limit the calculation of the area under the Xth percentile | should be bit bigger than the max width of the entrance
    'XB_intersection_search_distance' : 100,    #window on either side of the XB intersection to locate maximum (m)NDWI in the area of the channel bottleneck.
    
    # other parameters - do not have to be changed
    'satnames' : satnames,                      #these satellites are included in postprocessing. This variable is defined at the top of this script. 
    'satnames_string' : satnames_string,        #inherited from initial settings
    
    # plotting parameters - do not have to be changed
    'closed_color' : 'orangered',
    'open_color' : 'royalblue',
    'xaxisadjust' : 0,
    'satnames_XS' : satnames, #plot spectral transects for these satellites | inherited from initial settings
    'satname_img' : 'S2',  #use this satellite for an illustration RGB image - typically S2 | requires for this satellite to be downloaded as imagery
    'Interpolation_method' : "bicubic", # interpolate the RGB images for illustration - choose between "None" #"bicubic"
    'linestyle' : ['-', '--', '-.'],
    'labelsize' : 18,
    'linewidth' : 2 ,
    'markersize': 5   #size of blue and orange dots indicating open vs. closed entrance states in delta-to-median time series
    }


#load the pickle file containing the outputs of the pathfinding algorithm. This will load the results processed via the above 'settings_entrance' from Step 4

#define data input and figure output paths
postprocess_out_path = os.path.join(filepath_data, sitename,  'results_' + settings['inputs']['analysis_vrs'], 'XB' + str(settings_entrance['XB_cost_raster_amp_exponent']) + 
                                '_AB' + str(settings_entrance['AB_cost_raster_amp_exponent']) + '_' + settings_entrance['path_index'])
figure_out_path = os.path.join(postprocess_out_path,'analyzed_for_' + postprocess_params['satnames_string'] + '_' + postprocess_params['spectral_index']  + '_' +  postprocess_params['Postprocessing_version'])

#create directories if they do not already exist
if not os.path.exists(postprocess_out_path ):
        os.makedirs(postprocess_out_path) 
if not os.path.exists(figure_out_path):
        os.makedirs(figure_out_path) 

#open pickly file generated via SDS_entrance.automated_entrance_paths
infile = open(os.path.join(postprocess_out_path, sitename + '_entrance_lines_auto_' + settings_entrance['path_index'] +'_based_Loop_dict.pkl'),'rb')

#load indices along the transects from Step 4 into the cross-section dataframe (XS_df)
XS_dict = pickle.load(infile)
infile.close()
XS_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in XS_dict.items() ]))

#load spatial transects into the cross-section pandas geodataframe (XS_gdf)
infile = open(os.path.join(postprocess_out_path,  sitename + '_entrance_lines_auto_' + settings_entrance['path_index'] +'_based_Loop_gdf.pkl'),'rb')
XS_gdf = pickle.load(infile)
infile.close()

#load training data into dataframe
Training_data_df  =  pd.read_csv(glob.glob(os.path.join(filepath_data, sitename) + '/user_validation_data/*' +  '*training*' +  settings_training['username'] +  '.csv' )[0], index_col=0) 
Training_data_df = Training_data_df[~Training_data_df.index.duplicated(keep='first')] #remove possible duplicate entries. 




#%% ######################
# 5.2 Identify optimal classification threshold and classify the image series into binary entrance states
########################## 

#create dataframe of delta-to-median parameter for all open and closed training images as the basis for identifying the optimal classificaiton threshold
Classification_df = SDS_entrance.setup_classification_df(XS_df, Training_data_df,postprocess_params)

#identify optimal classification threshold: binary entrance states are defined as 'open' = 1, 'closed' =  0
Validation_stats_df={} 
Validation_stats_df[sitename + '_Across_berm'] = SDS_entrance.bestThreshold(Classification_df['user_entrance_state'],  Classification_df['Across_berm_DTM'])
Validation_stats_df[sitename + '_Along_berm'] = SDS_entrance.bestThreshold(Classification_df['user_entrance_state'],  Classification_df['Along_berm_DTM'])
Validation_stats_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in Validation_stats_df.items() ])) 
Validation_stats_df = Validation_stats_df.transpose()
Validation_stats_df.columns = ['Fscore','Accuracy', 'TN', 'FP', 'FN', 'TP', 'Opt_threshold']  #TN = true negative, FP = false positives... 
print('Classification accuracy metrics are: ')
print(Validation_stats_df)
print('')

#set the thresholds for along and across-berm classification into binary entrance states
Analysis_direction =  'AB' #AB for along-berm, XB for across berm. Choose the analysis direction that lead to the highes Fscore and Accuracy above. 

#use optimal threshold inferred from user validation data - the parameter is DTM_threshold
if Analysis_direction == 'XB':
    DTM_threshold =  Validation_stats_df['Opt_threshold'][0] #if across-berm classification stats were better 
else:
    DTM_threshold =  Validation_stats_df['Opt_threshold'][1] #if along-berm classification stats were better 

#Alternatively, use a user-defined fixed threshold via overriding of the DTM_threshold parameter established above. 
#DTM_threshold =  0.1

#classify the full image series based on the best performing analysis direction and corresponding threshold
XS_DTM_classified_df = SDS_entrance.classify_image_series_via_DTM(XS_df, Analysis_direction, DTM_threshold, postprocess_params)


#%% ######################
#5.3 Check detection
##########################

#Here, the user has the ability to go over each automated entrance state detection via an interactive pop-up window
#this can be used to get rid of possibly cloudy or otherwise problematic detections - to ultimately create a clean result time series 
#Note that you have to go through the entire image series for this step 
#For a first pass assessment or under limited time, skip this step! 

#run the 'check detection' function and create a clean XS_DTM_classified_df
XS_DTM_classified_df = SDS_entrance.check_entrance_state_detection(postprocess_out_path, XS_DTM_classified_df.iloc[0:10])
    
    
    
#%% ######################
#5.4 Divide the original data into open and closed entrance states and plot result figures
########################## 

#Create dataframes and geodataframes corresponding to the open and closed classified entrance states. 
#These are used for creating the standart plots but can serve as the basis for any additional user-specific analysis

XS_o_df = pd.DataFrame()       #dataframe containing transect data for all 'open' entrance states
XS_o_gdf = pd.DataFrame()      #geodataframe containing trasects for all 'open' entrance states
for date in XS_DTM_classified_df[XS_DTM_classified_df['bin_entrance_state'] == 1].index:
    XS_o_df = pd.concat([XS_o_df, pd.DataFrame(XS_df.filter(regex=date))], axis = 1)
    XS_o_gdf = pd.concat([XS_o_gdf, XS_gdf[XS_gdf['date'] == date]], axis = 0)

#write open across-berm transects out as shapefile for visualization
XS_o_gdf[XS_o_gdf['direction']=='XB'].to_file(figure_out_path  + '/open_entrance_along_berm_paths.shp', driver='ESRI Shapefile')     

XS_c_df = pd.DataFrame()        #dataframe containing transect data for all 'open' entrance states
XS_c_gdf = pd.DataFrame()       #geodataframe containing transects for all 'open' entrance states
for date in XS_DTM_classified_df[XS_DTM_classified_df['bin_entrance_state'] == 0].index:
    XS_c_df = pd.concat([XS_c_df, pd.DataFrame(XS_df.filter(regex=date))], axis = 1)
    XS_c_gdf = pd.concat([XS_c_gdf, XS_gdf[XS_gdf['date'] == date]], axis = 0)

#plot result figures and save processed data as csv
SDS_entrance.plot_entrancesat_results(XS_o_df, XS_c_df,XS_o_gdf, XS_c_gdf, settings, postprocess_params, metadata,  figure_out_path)








        
    
        