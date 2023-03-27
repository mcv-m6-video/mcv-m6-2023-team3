import xml.etree.ElementTree as ET
import pandas as pd

# Parse the XML file
name_file = "ai_challenge_s03_c010-full_annotation"
tree = ET.parse(name_file+'.xml')
root = tree.getroot()

# create a list of dictionaries to store the data
data = []
for track in root.findall('track'):
    for box in track.findall('box'):
        data.append({
            'frame': box.get('frame'),
            'xtl': box.get('xtl'),
            'ytl': box.get('ytl'),
            'xbr': box.get('xbr'),
            'ybr': box.get('ybr'),
            'occluded': box.get('occluded'),
            'parked': box.find('attribute').text if box.find('attribute') is not None else ''
        })

# create a pandas dataframe from the data list
df = pd.DataFrame(data)

# save the dataframe to a CSV file
df.to_csv(name_file+'.csv', index=False)