# EntranceSat

Valentin Heimhuber, University of New South Wales, Water Research Laboratory, 02/2021


## **Description**

EntranceSat is a Google Earth Engine enabled open source python software package that first uses a novel least cost path finding approach to trace IOCE entrance channels along and across the berm, and then analyses the resulting transects to infer whether an entrance is open or closed. EntranceSat is built on the satellite imagery download and pre-processing functionality of the CoastSat toolbox.

![Alt text](readme_files/EntranceSat_method_illustration.gif)

The underlying approach of the EntranceSat toolkit and it's performance is described in detail in the following publication:

Valentin Heimhuber, Kilian Vos, Wanru Fu, William Glamore (submitted): EntranceSat: A Google Earth Engine enabled open-source python tool for historic and near real-time monitoring of intermittent estuary entrances

Output files of EntranceSat include:
- Time series of along-berm and across-berm paths as .shp file for use in GIS software (no tidal correction)
- NIR, SWIR1, NDWI and mNDWI extracted along each along-berm and across-berm path
- For each processed image, the tool will output a 'dashboard' showing the location of the along-berm and across-berm paths along with the user-defined spectral transects and the level of the tide
- A variety of timeseries plots that illustrate the results of the algorithm


## **Installation**

The EntranceSat toolkit is run in the original CoastSat python environment, which uses the Google Earth Engine API to download and pre-process the satellite imagery. Please Refer to the CoastSat installation instructions 1.1. [https://github.com/kvos/CoastSat] for guidance on how to set up this environment correctly.

The basic steps to do this are: Download this repository -> Download the environment.yml file from the CoastSat repository and place it in the EntranceSat directory -> Download Anaconda -> create the EntranceSat environment via conda env create -f environment.yml -n EntranceSat -> 1.2 Activate Google Earth Engine Python API via CoastSat instructions -> you should be good to go.


Additional packages to manually install in the coastsat environment are:
- Rasterio [pip install rasterio]
- fbriol fes [conda install -c fbriol fes]
- glob [conda install glob2]
- seaborn [conda install seaborn]


ICOLLsat uses a global tide model reanalysis dataset provided through the pyfes package and this cannot just simply be installed through anaconda. FES2014 ranks as one of the best global tide models for coastal areas (based on an assessment on 56 coastal tide gauges, see this paper https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014RG000450.



	How to install fbriol fes (the global tide model (FES)) in a python environment:

	1. Download the source folder from here: https://bitbucket.org/fbriol/fes/downloads/ - unzip it and store in any prefered location.
	2. run: `conda install -c fbriol fes` #this works without step 1
	3. Test `import pyfes` in python
	4. Download **ocean tide**, **ocean_tide_extrapolated** and **load tide** .nc files from Aviso+ [gmail account, password: OzjEB5] or ask Kilian or Tino for the files - these are large netcdf files that contain all the outputs from the global tide model
	5. Save the .nc files in the source folder, under /data/fes2014 - when using fes in python, it will look for these files in the source folder, but the filepaths to each file have to be given manually (see code below)
	6. Update the .ini file to include the absolute path to each tidal component (.nc file)
	Done

	7. Using pyfes: below is a python function for extracting tide water levels for a given lat lon location and a vector of datetime64 formatted dates.
	import pyfes
	import datetime
	def compute_tide_dates(Tidelatlon,dates):
		config_ocean_extrap = r"H:\Downloads\fes-2.9.1-Source\data\fes2014\ocean_tide_extrapolated.ini"
		ocean_tide = pyfes.Handler("ocean", "io", config_ocean_extrap)
		config_load = r"H:\Downloads\fes-2.9.1-Source\data\fes2014\load_tide.ini"
		load_tide = pyfes.Handler("radial", "io", config_load)
		'compute time-series of water level for a location and dates (using a dates vector)'
		dates_np = np.empty((len(dates),), dtype='datetime64[us]')
		for i,date in enumerate(dates):
		dates_np[i] = date #datetime(date.year,date.month,date.day,date.hour,date.minute,date.second) #if dates is already datetime64 format - otherwise use the long version here
		lons = Tidelatlon[0]*np.ones(len(dates))
		lats = Tidelatlon[1]*np.ones(len(dates))
		# compute heights for ocean tide and loadings
		ocean_short, ocean_long, min_points = ocean_tide.calculate(lons, lats, dates_np)
		load_short, load_long, min_points = load_tide.calculate(lons, lats, dates_np)
		# sum up all components and convert from cm to m
		tide_level = (ocean_short + ocean_long + load_short + load_long)/100
		return tide_level

## **Data Requirements**

All user input files (area of interest polygon, transects & tide data) should be saved in the folder "...CoastSat.PlanetScope/user_inputs"
- Analysis region of interest .kml file can be selected and downloaded from [geojson.io].
- Transects .geojson file (optional) should match the user input settings epsg. If skipped, transects may be drawn manually with an interactive popup. Alternately, the provided NARRA_transect.geojson file may be manually modified in a text editor to add/remove/update transect names, coordinates and epsg
- Tide data .csv for tidal correction (optional) should be in UTC time and local mean sea level (MSL) elevation. See NARRA_tides.csv for csv data and column name formatting.

Beach slopes for the tidal correction (step 5.) can be extracted using the CoastSat.Slope toolkit [https://github.com/kvos/CoastSat.slope]


## **Usage**

![](readme_files/timeseries.png)

It is recommended the toolkit be run in spyder. Ensure spyder graphics backend is set to 'automatic' for proper plot rendering.
- Preferences - iPython console - Graphics - Graphics Backend - Automatic

CoastSat.PlanetScope is run from the CoastSat_PS.py file.
- Instructions and comments are provided in this file for each step.
- It is recommended steps be run as individual cells for first time users.

Settings and interactive steps are based on the CoastSat workflow and will be familiar to users of CoastSat.

Interactive popup window steps include:
- Raw PlanetScope reference image selection for co-registration [step 1.2.]
- Top of Atmosphere merged reference image selection for shoreline extraction [step 2.1.]
- Reference shoreline digitisation (refer 'Reference shoreline' section of CoastSat readme for example) - [step 2.1.]
- Transect digitisation (optional - only if no transects.geojson file provided) - [step 2.1.]
- Manual error detection (optional - keep/discard popup window as per CoastSat) - [step 3.]


## **Training Neural-Network Classifier**

Due to the preliminary stage of testing, validation has only been completed at Narrabeen-Collaroy beach (Sydney, Australia). As such, the NN classifier is optimised for this site and may perform poorly at sites with differing sediment composition. It is recommended a new classifier be trained for such sites.

Steps are provided in "...CoastSat.PlanetScope/coastsat_ps/classifier/train_new_classifier.py".
- Instructions are in this file and based of the CoastSat classifier training methods [https://github.com/kvos/CoastSat/blob/master/doc/train_new_classifier.md].
- CoastSat.PlanetScope must be run up to/including step 1.3. on a set of images to extract co-registered and top of atmosphere corrected scenes for classifier training.


## **Validation Results**

- Accuracy validated against in-situ RTK-GPS survey data at Narrabeen-Collaroy beach in the Northen beaches of Sydney, Australia with a RMSE of 3.66m (n=438).
- An equivelent validation study at Duck, North Carolina, USA provided an observed RMSE error of 4.74m (n=167).


Detailed results and methodology outlined in:

Doherty Y., Harley M.D., Vos K., Splinter K.D. (2021). Evaluation of PlanetScope Dove Satellite Imagery for High-Resolution, Near-Daily Shoreline Monitoring (in peer-review)
# EntranceSat
