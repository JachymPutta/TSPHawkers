###########################################################################
# TSP 
###########################################################################
import csv
import googlemaps
import math
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from python_tsp.heuristics import solve_tsp_simulated_annealing
import plotly.graph_objects as go

#Optimal solution
#from python_tsp.exact import solve_tsp_dynamic_programming

api_key = "API key" 
gmaps = googlemaps.Client(key=api_key)

################################################################################
# Getting the data initially 
################################################################################


# res = pd.DataFrame(index = hknm,columns= hknm)

# get_distance_matrix(hk_names)
# res = res.fillna(0)

#Saving the distance matrix to a csv file
#res.to_csv('hawker_distance_matrix.csv', index = False)

#Save the raw data 
# with open('raw_data.txt', 'w') as filehandle:
#     for listitem in raw_data:
#         filehandle.write('%s\n' % listitem)  


################################################################################
# Importing raw data
################################################################################
rawDataFile = 'raw_data.txt'
distanceMatrixFile = 'hawker_distance_matrix.csv'
hknamesFile = 'hawkers.csv'

def flatten(lst):
    flat_list = [item for sublist in lst for item in sublist]
    return flat_list

def get_distance_matrix(lst, ind = 0):
    for i in range(ind,len(lst)):
        for j in range(i):
           step = gmaps.distance_matrix(lst[i],lst[j], language = 'English')
           raw_data.append(step)
           res.iloc[i,j] = step["rows"][0]["elements"][0]['duration']["value"]


raw_data = []
with open(rawDataFile, 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        raw_data.append(currentPlace)

hk_names = []
with open(hknamesFile, mode= 'r') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        hk_names.append(row)
        hk_names[0] = ['Adam Food Centre']

res = pd.read_csv(distanceMatrixFile, index_col = False)

#Distances in meters
dist = []    
for e in raw_data:
    x = re.findall(r"'value': \d+",str(e))
    j = [int(s) for s in x[0].split() if s.isdigit()]
    dist.append(j)
dist = flatten(dist)

distances_meters = np.zeros((len(hk_names),len(hk_names)))
for i in range(len(hk_names)):
    for j in range(i):
        distances_meters[i,j] = dist[i+j]


###########################################################################
# Minimum spanning tree 
###########################################################################

#Mst - seconds
s_matrix = csr_matrix(res.values)
Tcsr = minimum_spanning_tree(s_matrix)
Tcsr.toarray().astype(int)
np.sum(Tcsr)
#MST size = 34634s ~ 9.62hr

#Mst - meters
m_matrix = csr_matrix(distances_meters)
Tcsr = minimum_spanning_tree(m_matrix)
Tcsr.toarray().astype(int)
np.sum(Tcsr)
#MST size = 61582m ~ 61.5km


###########################################################################
# Travelling salesman
###########################################################################

dist_matrix_seconds = np.array(res.values) + np.transpose(np.array(res.values))
dist_matrix_meters = distances_meters + distances_meters.transpose()


#Optimal solution - global minimum
#permutation, distance = solve_tsp_dynamic_programming(dist_matrix)

#Heuristic solution - checking local minima

#TSP - Seconds
permutation, distance = solve_tsp_simulated_annealing(dist_matrix_seconds)
#TSP - Meters
permutation, distance = solve_tsp_simulated_annealing(dist_matrix_meters)

#Distance(seconds) = 43585s ~ 121 hours

################################################################################
#Custom ordering for the nodes
################################################################################

hk_lat_lon = pd.read_csv('hk_lat_lon.csv', index_col = False)

route = []
for x in permutation:
    route.append(hk_names[x])
route = flatten(route)

sorterIndex = dict(zip(route, range(len(route))))

hk_lat_lon['Rank'] = hk_lat_lon['Name'].map(sorterIndex)
hk_lat_lon.sort_values(['Rank'], ascending = True, inplace = True)
hk_lat_lon.drop('Rank', 1, inplace = True)

################################################################################
# Visualize
################################################################################

mapbox_access_token = "key"
#Plotting the route
fig = go.Figure(go.Scattermapbox(lat= hk_lat_lon['Latitude'],
        lon= hk_lat_lon['Longitude'],
        mode='markers+text+lines',
        marker=go.scattermapbox.Marker( size=5 ),
        text = hk_lat_lon['Name'],
        textfont=dict(size=8, color='black')))

fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=1.3351141020360606, 
            lon=103.82763491866139
        ),
        pitch=0,
        zoom=10
    ),
)

fig.show()

#Exporting the image
#fig.write_image('allhk.png', width=800, height=600, scale=1)

