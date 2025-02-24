

import os
import math
import requests
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import MultiPolygon, Polygon, box
from PIL import Image


csv_file_path = r'C:\Users\Raissa\Documents\dev\TCC\AnaliseExploratoria\chicago_tracts_boundaries.csv'#r'C:\Users\Raissa\Documents\dev\TCC\AnaliseExploratoria\PRECINCTS_2012_20240603.csv'
api_key = os.environ.get('Google_Maps_Tiles_API')
output_folder = 'chicago_satellite_tracts_images_large'
output_txt_file = 'chicago_satellite_tracts_images.txt'
img_size = 1280 # 256 
output_csv = 'chicago_satellite_tracts_images_precinct.csv'

os.makedirs(output_folder, exist_ok=True)

def get_centroid(polygon):
    return polygon.centroid.x, polygon.centroid.y

def rectangle_centroid(rectangle):
    return rectangle.centroid.x, rectangle.centroid.y


def divide_polygon_into_grid(polygon):

    minx, miny, maxx, maxy = polygon.bounds
    

    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2

    rect1 = box(minx, midy, midx, maxy)  
    rect2 = box(midx, midy, maxx, maxy)  
    rect3 = box(minx, miny, midx, midy)  
    rect4 = box(midx, miny, maxx, midy)  

    centroids = [rectangle_centroid(rect) for rect in [rect1, rect2, rect3, rect4]]
    
    return centroids


def fetch_and_save_satellite_image(lat, lon, zoom):

    api_key = os.environ.get('Google_Maps_Tiles_API')
    img_size = 1280 # 256 

    tile_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&scale=2&size={img_size}x{img_size}&maptype=satellite&format=png&visual_refresh=true&key={api_key}" #&size=1280x1280
    )
    response = requests.get(tile_url)
    print(response.status_code)
    if response.status_code == 200:
        tile_image = Image.open(requests.get(tile_url, stream=True).raw)

    else:
        print(f"Erro {response.status_code} ao baixar tile em x: {lat}, y: {lon}, zoom: {zoom}")


    image_filename = f"satellite_{lat}_{lon}_zoom{zoom}.png"
    image_path = os.path.join(output_folder, image_filename)
    tile_image.save(image_path)

    return image_filename


#image_filename = fetch_and_save_satellite_image(  41.85,-87.65,17)

zoom_level = 17
if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
else:
    df = pd.read_csv(csv_file_path)
    df['saved_images'] = None  


for index, row in df.iterrows():

    if pd.notna(row['saved_images']):
        continue  


    multipolygon = loads(row['geometry_coordinates'])
    row_data = row.drop('geometry_coordinates').to_dict()
    image_filenames = []


    if isinstance(multipolygon, MultiPolygon):
        for polygon in multipolygon.geoms:
            centroids = divide_polygon_into_grid(polygon)
            for lon, lat in centroids:

                image_filename = fetch_and_save_satellite_image(lat, lon, zoom_level)
                image_filenames.append(image_filename)
             


    elif isinstance(multipolygon, Polygon):
        centroids = divide_polygon_into_grid(multipolygon)
        for lat, lon in centroids:
            image_filename = fetch_and_save_satellite_image(lat, lon, zoom_level)
            image_filenames.append(image_filename)


    df.at[index, 'saved_images'] = str(image_filenames)

    df.to_csv(output_csv, index=False)
