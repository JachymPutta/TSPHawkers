# TSPHawkers
Travelling salesman problem solver based on google maps API

The following implementation is based on the Google Maps API for gathering data, as well as mapbox API for creating the graphs. This means that to run the code on a new dataset, both API keys are required. 

The Google API key is the 'api_key' variable. Conversely the mapbox is under the 'key' variable. 

This program provides a simple way to solve the travelling salesman problem based on data pulled directly from Google maps. More details on the functions can be found here: https://developers.google.com/maps/documentation/distance-matrix/overview

The functionality is demonstrated on 107 hawker centers in Singapore. The size of the input forces the use of a heuristic to compute the solution. The function used to find the global optimum is provided in the comments below.
