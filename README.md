# EntranceSat

Valentin Heimhuber, University of New South Wales, Water Research Laboratory, 02/2021


## **Description**

EntranceSat is a Google Earth Engine enabled open source python software package that first uses a novel least cost path finding approach to trace IOCE entrance channels along and across the berm, and then analyses the resulting transects to infer whether an entrance is open or closed. EntranceSat is built on top of the imagery download and pre-processing functionality of the CoastSat toolbox [https://github.com/kvos/CoastSat].

![Alt text](readme_files/EntranceSat_method_illustration.gif)

The underlying approach of the EntranceSat toolkit and it's performance is described in detail in the following publication:

Valentin Heimhuber, Kilian Vos, Wanru Fu, William Glamore (submitted): EntranceSat: A Google Earth Engine enabled open-source python tool for historic and near real-time monitoring of intermittent estuary entrances

Output files of EntranceSat include:
- Time series of along-berm and across-berm paths as .shp file for use in GIS software (no tidal correction)
- NIR, SWIR1, NDWI and mNDWI extracted along each along-berm and across-berm path
- For each processed image, the tool will output a 'dashboard' showing the location of the along-berm and across-berm paths along with the spectral transects and level of the tide (if the tide model is integrated)
- A variety of timeseries plots that illustrate the key results of the algorithm


## 1. Installation

The EntranceSat toolkit is run in the original CoastSat python environment, with a few additional python packages added to it. It uses the Google Earth Engine API to download and pre-process the satellite imagery based on the corresponding functions of CoastSat [https://github.com/kvos/CoastSat]. The installation instructions provided here are replicated from CoastSat and only slightly modified here.  

### 1.1 Download repository

Download this repository from Github and unzip it in a suitable location.

### 1.2 Create a python environment with Anaconda

To run the toolbox you first need to install the required Python packages in an environment. To do this we will use **Anaconda**, which can be downloaded freely [here](https://www.anaconda.com/download/).

Once you have it installed on your PC, open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command (change directory) to go the folder where you have downloaded this repository. Copy the filepath to that folder and replace the below filepath with that.

```
cd C:\Users\EntranceSat
```

Create a new environment named `entrancesat` with all the required packages:


```
conda env create -f environment.yml -n entrancesat
```

All the required packages have now been installed in an environment called `entrancesat`. Now, activate the new environment:

```
conda activate entrancesat
```

To confirm that you have successfully activated entrancesat, your terminal command line prompt should now start with (entrancesat).

**In case errors are raised:**: open the **Anaconda Navigator**, in the *Environments* tab click on *Import* and select the `environment.yml` file. For more details, the following [link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) shows how to create and manage an environment with Anaconda.

### 1.3 Activate Google Earth Engine Python API

First, you need to request access to Google Earth Engine at https://signup.earthengine.google.com/. It takes about 1 day for Google to approve requests.

Once your request has been approved, with the `entrancesat` environment activated, run the following command on the Anaconda Prompt to link your environment to the GEE server:

```
earthengine authenticate
```

A web browser will open, login with a gmail account and accept the terms and conditions. Then copy the authorization code into the Anaconda terminal.

Now you are ready to start using the EntranceSat toolbox!

**Note**: remember to always activate the environment with `conda activate entrancesat` each time you are preparing to use the toolbox. Once you do this from the Annaconda command prompt, you can startup Spyder simply by typing spyder in the command line.


### 1.4 Install and activate the FES global tide model (for advanced python users)

ICOLLsat uses the FES2014 global tide model reanalysis dataset to estimate the level of the tide for the point in time where each satellite image was acquired, since this tide level determines the depth of the water column in open entrances. FES2014 ranks as one of the best global tide models for coastal areas (based on an assessment on 56 coastal tide gauges, see this paper https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014RG000450.

Importantly, this information is provided to the user but it is not currently used in any form of calculation or the determination of open vs. closed entrance states. Since the setup of the tide model is rather technical, some users might choose to not incorporate this data into the anlaysis, which is an option. Below are the steps required to setup the FES2014 tide model integration for EntranceSat.


	How to install fbriol fes (the global tide model (FES)) in a python environment:

	1. Download the source folder from here: https://bitbucket.org/fbriol/fes/downloads/ - unzip it and store in any preferred location.
	2. run: `conda install -c fbriol fes` #this works without step 1
	3. Test `import pyfes` in python
	4. Download **ocean tide**, **ocean_tide_extrapolated** and **load tide** .nc files from Aviso+
	as explained here https://bitbucket.org/fbriol/fes/src/master/README.md.
	These are large (8gb+) netcdf files that contain all the results from the global tide model. They are not included in the Installation
	of fbriol fes via conda install and hence, have to be downloaded separately and placed in the correct locations.
	5. Save the .nc files in the source folder, under /data/fes2014 (e.g. data\fes2014\ocean_tide_extrapolated\).
	6. When using fes in python, it will look for these files in the source folder, but the filepaths to each file have to be provided explicitly (manually) in the. Update the ocean_tide_extrapolated.ini file to include the absolute path to each tidal component (.nc file)
	Done


### Using EntranceSat

#### Setting up the algorithm for processing**

All user input files (area of interest polygon, transects & tide data) are kept in the folder ".../Entrancesat/user_inputs"

It is recommended that new analysis regions/ IOCEs are added directly to the input_locations.shp file via QGIS or ArcGIS. For each site, EntranceSat expects 6 polygons as shown in this example. In the attribute table of this shapefile, each of these 8 unique polygons has to be named accordingly in the 'layer' field. Each polygon also requires the sitename in the 'sitename' field.

![Alt text](https://github.com/VHeimhuber/EntranceSat/blob/main/readme_files/EntranceSat%20site%20setup%20illustration.jpg)

The full_bounding_box polygon is used for selecting and cropping of the satellite imagery from the Google Earth Engine. This does not necessarily have to incorporate the entire estuary. The area of this polygon should not exceed 100 km2

Points A-D should be kept as very small triangles. The first point of this triangle is used as seed/receiver point during analysis. This is a workaround since ESRI shapefiles cannot contain polygons and points at the same time.

The A-B and C-D masks are used for limiting the area for least-cost pathfinding to within those polygons. The location of the seed and receiver points and shape of these masks can significantly affect the performance of the pathfinding and it is recommended to play around with these shapes initially by doing test runs based on a limited number of images until satisfactory performance is achieved.


#### Running the tool in python

It is recommended the toolkit be run in spyder. Ensure spyder graphics backend is set to 'automatic' for proper plot rendering.
- Preferences - iPython console - Graphics - Graphics Backend - Automatic

EntranceSat is run from the EntranceSat_master.py file.
- Instructions and comments are provided in this file for each step.
- It is recommended steps be run as individual cells for first time users.

The key steps are briefly outlined here:

-At the top of the script, enter the sitename of your site and ensure it matches the spelling used in the input_locations.shp file. The code will then extract the polygons from this shapefile for your site.
-Choose the time period for analysis and the satellites you want to include.
-Download the imagery via metadata = SDS_download.retrieve_images(inputs)
-Generate training data. Generally, the more the better but we recommend at least 10 open and 10 closed entrance states to be included
for each group of satellites (Landsat 5,7 and 8 (1 group) and Sentinel-2). Use the arrows on the keyboard to 'classify' each image. Equal 'class sizes' can be achieved by using 'skip'.

![Alt text](https://github.com/VHeimhuber/EntranceSat/blob/main/readme_files/training_data_generator.png)

-Run the least-cost path finding step. This is the core step of the algorithm. This will process all downloaded images that have less than the user defined threshold of cloud cover over the across-berm mask. For each image, the least-cost path finding algorithm is run and the core bands and spectral indices are then extracted along those transects as input to the second major analysis step. Most of the input parameters to the SDS_entrance.automated_entrance_paths function don't need to be changed. The first 8 parameters should be considered more carefully.  

For each image, EntranceSat creates the following 'dashboard' output as a png file.
![Alt text](https://github.com/VHeimhuber/EntranceSat/blob/main/readme_files/1989-02-20-23-14-27_L5_DURRAS_swir_based_ABexp_%2025_XBexp_%2025_MAdis_60.png)

-Inferring open vs. closed entrance states:

Based on the data generated by the path finding algorithm step, EntranceSat then calculates the delta-to-median parameter for each image, which is indicative of the state of the entrance. The calculation of this parameter is explained in the following figure. Details are provided in the publication listed above and in the code itself.

![Alt text](https://github.com/VHeimhuber/EntranceSat/blob/main/readme_files/inferring entrance states method illustration.png)

Based on the user generate training data, the next step then is to identify the optimal classification threshold for distinguishing between open and closed entrance states. This is done automatically via the SDS_entrance.bestThreshold function.

After this threshold is identified, the full image series is then classified into open vs. closed entrance states and a number of output datasets are generated in a dedicated directory. These outputs include:

-Location plots of the identified least cost paths:
![Alt text](https://github.com/VHeimhuber/EntranceSat/blob/main/readme_files/Figure_1_mndwi_11-02-2021_L5_L7_L8_S2.png)
-Time series plots of the delta to median parameter:
![Alt text](https://github.com/VHeimhuber/EntranceSat/blob/main/readme_files/Figure_2_mndwi_11-02-2021_L5_L7_L8_S2.png)
-CSV files with all the processed data.
-Shapefile containing all of the least-cost paths corresponding to open entrances.
