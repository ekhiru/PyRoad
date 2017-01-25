import numpy as np
from itertools import islice
from imp import reload
import pandas as pd
import json

import RG_clean as cl
from imp import reload
reload(cl)

def read_road_data(file_road):
    with open(file_road) as data_file:
        data = json.load(data_file)
    coords = []
    segments = {}
    tagslist = []
    for place in data['elements']:
        if place['type']=='node':
            coords.append(place)
            # return place
        if place['type']=='way':
            dfpoints = pd.DataFrame(columns=['p_id', 'way_id'])
            dfpoints['p_id'] = place['nodes']
            way_id = str(place['id'])
            dfpoints['way_id'] = way_id
            if way_id not in segments:
                segments[way_id] = dfpoints
            else:
                print("alfo pasa")
            tags = place['tags']
            tags['way_id'] = way_id
            tags['pt_ini'] = place['nodes'][0]
            tags['pt_end'] = place['nodes'][-1]
            tagslist.append(tags)
    dfways = pd.DataFrame(tagslist)
    if 'junction' not in dfways.columns:
        dfways['junction'] = None
    dfcoord = pd.DataFrame(coords)
    dfcoord = dfcoord.drop([i for i in ['type','tags'] if i in dfcoord.columns ], axis=1)
    df = dfcoord.groupby(['id']).size().reset_index(name='repes')
    dfcoord = pd.merge(dfcoord, df, left_on='id', right_on='id')
    dfcoord.repes = dfcoord.repes.astype(int)
    dfcoord = dfcoord.drop_duplicates()
    return dfways, segments, dfcoord


colors_code={4: '9f000000', 3: '9f0000FF', 2: '9f0088FF', 1: '9f00FFFF', 0: '9f00FF00',
    5: '9fFFFFFF' , #white
    6: '9f9FF897' ,#blando y verde
    7: '9f14F0CF', # verde amarillo
    8: '9f14C8FF',  #amarillo naranja
    9: '9f1461FF' , # rananja rojo
    10: '9f0A0084',  #rojo negro
    11: '11FF78F0',  # rosa
    12: '9f14005A', #marron

    20	:'ffe3cea6'	,
    21	:'ffb4781f'	,
    22	:'ff8adfb2'	,
    23	:'ff2ca033'	,
    24	:'ff999afb'	,
    25	:'ff1c1ae3'	,
    26	:'ff6fbffd'	,
    27	:'ff007fff'	,
    28	:'ffd6b2ca'	,
    29	:'ff9a3d6a'	,
    30	:'ff99ffff'	,
    31	:'ff2859b1'	,
    32	:'ffc7d38d'	,
    33	:'ffb3ffff'	,
    34	:'ffdababe'	,
    35	:'ff7280fb'	,
    36	:'ffd3b180'	,
    37	:'ff62b4fd'	,
    38	:'ff69deb3'	,
    39	:'ffe5cdfc'	,
    40	:'ffd9d9d9'	,
    41	:'ffbd80bc'	,
    42	:'ffc5ebcc'	,
    43	:'ff6fedff'
    }
    # http://colorbrewer2.org/#type=qualitative&scheme=Set3&n=12
    # http://eartz.github.io/rgbaToKml/




def write(out_path, dfs, max_lat, min_lat, max_lon, min_lon, dfseg=None):
    # 0000FF00 verde
    # 00FF0000 azul
    # 000000FF rojo
    # 00FFFF00 azul clarito
    # 0000FFFF amarillo
    # 000088FF naranja
    # 5-azul,4-negra, 3-roja, 2-naranja, 1-amarillo, 0-verde
    f = open(out_path, 'w')
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n')
    f.write('<Document>\n')

    for i in colors_code.keys():
        f.write('<Style id="lineStyle'+str(i)+'">\n')
        f.write('\t<LineStyle>\n')
        f.write('\t\t<color>'+colors_code[i]+'</color>\n')
        f.write('\t\t<width>4</width>\n')
        f.write('\t</LineStyle>\n')
        f.write('</Style>\n')
    for i in range(1,11):
        f.write('<Style id="marker_sty_'+str(i/10).replace(".", "_")+'">\n')
        f.write('\t<IconStyle>\n')
        f.write('\t\t<color>ddffffff</color>\n')
        f.write('\t\t<colorMode>normal</colorMode>\n')
        f.write('\t\t<scale>'+str(i/10+0.2)+'</scale>\n') #+0.2 para hacerlos un poco mas grandes
        f.write('\t</IconStyle>\n')
        f.write('</Style>\n')
    f.write('<Style id="marker_sty_red">\n')
    f.write('\t<IconStyle>\n')
    f.write('\t\t<color>dd0000ff</color>\n')
    f.write('\t\t<colorMode>normal</colorMode>\n')
    f.write('\t\t<scale>1</scale>\n') #+0.2 para hacerlos un poco mas grandes
    f.write('\t</IconStyle>\n')
    f.write('</Style>\n')


    f.write('<LatLonBox>\n')
    f.write('\t<north>' + str(max_lat) + '</north>\n') #lat max
    f.write('\t<south>' + str(min_lat) + '</south>\n')
    f.write('\t<east>' + str(max_lon) + '</east>\n') #lon max
    f.write('\t<west>' + str(min_lon) + '</west>\n')
    f.write('</LatLonBox>\n')
    # for df in dfs:
    f.write('<Style id="folderStyle">\n')
    f.write('\t<ListStyle>\n')
    f.write('\t\t<listItemType>checkHideChildren</listItemType>\n')
    f.write('\t</ListStyle>\n')
    f.write('</Style>\n')


    for df in dfs:
        df['color'] = pd.cut(df.curv_radius, bins=cl.radius_discret_edges, labels=cl.radius_discret_labels)
    write_color(f, dfs, 'Curvature', '')
    for df in dfs:
        df.loc[df.junction=='link', 'color']=4
        df.loc[df.junction=='roundabout', 'color']=3
        df.loc[df.junction=='intersect', 'color']=2
        df.loc[df.junction=='nojunct', 'color']=0
    write_color(f, dfs, 'Junction','')
    if 'seg_id' in dfs[0].columns:
        for df in dfs:
            df.loc[df.tunnel==True, 'color']=3
            df.loc[df.tunnel==False, 'color']=0
        write_color(f, dfs, 'Tunnel','')
    else: print("no info for tunnels")
    if dfseg is not None:
        write_segment_info (f, dfseg, dfs )
    else:
        print("No info for segments")
    if 'seg_id' in dfs[0].columns:
        for df in dfs:
            df.loc[df.seg_id.notnull(),'color']= 20+(df.seg_id % (len(colors_code)-20))
            df.loc[df.seg_id.isnull(),'color']= None
        write_color(f, dfs, 'Segments','')
    else: print("No info for seg_id")

    f.write('</Document>\n')
    f.write('</kml>\n')
    f.close()


def write_color(f, dfs, fol_name, fol_des):
    f.write('<Folder>\n')
    f.write('<styleUrl>#folderStyle</styleUrl>')
    f.write('<name>'+fol_name+'</name>\n')
    f.write('<description>'+fol_des+'</description>\n')
    for dfseg in dfs: # paths
        dfseg.color.fillna(5, inplace=True)
        dfseg.color=dfseg.color.astype("int")
        # df=dfseg[:-1]
        df=dfseg
        first = ~(df.color == df.color.shift()) # problems with NaN??
        last  = ~(df.color == df.color.shift(-1))
        for c, fi, l, lat, lon, late, lone in zip(df.color, first, last, df.ini_lat, df.ini_lon, df.end_lat, df.end_lon):
            if len(df)==1:
                fi=l=True
            if fi:
                if int(c) < 0 or int(c) > max(colors_code.keys()):
                    print("input_output::write_color: color error - ",fol_name,c, fi, l, lat, lon, late, lone)
                    print(df.tail())
                    print("lens", len(df), len(dfseg))
                    c = max(colors_code.keys())
                f.write('<Placemark>\n')
                f.write('\t<styleUrl>#lineStyle'+str(int(c))+'</styleUrl>\n')
                f.write('\t<LineString>\n')
                f.write('\t<extrude>1</extrude>\n')
                f.write('\t\t<tessellate>1</tessellate>\n')
                f.write('\t\t<coordinates>\n')
            f.write(str(lon)+','+str(lat)+' ')
            if l :
                if str(lone)=='nan':
                    print("input_output::write_color: NaN latitud")
                    # print(df.head())
                f.write(str(lone)+','+str(late)+' ')
                f.write('\n\t\t</coordinates>\n')
                f.write('\t</LineString>\n')
                f.write('</Placemark>\n')
    f.write('</Folder>\n')

def write_segment_info (f, dfseg, dfs ):
    # dfseg.count_lines	,dfseg.curv_rad_min	,dfseg.curv_rad_avg,	dfseg.length
    # coger un pto random de todos los del segmento (para no pintar en las esquinas)
    dfr = pd.concat(dfs).reset_index(drop=True)[['seg_id', 'ini_lat','ini_lon']]
    dfr = dfr.iloc[np.random.permutation(len(dfr))].reset_index(drop=True)
    dfseggeo = dfr.drop_duplicates(subset='seg_id')
    dfseg = pd.merge(dfseg, dfseggeo, on='seg_id')
    f.write('<Folder>\n')
    f.write('<name>seg_id_info</name>\n')
    for la, lo, seg,a,b in zip( dfseg.ini_lat,dfseg.ini_lon, dfseg.seg_id, dfseg.curv_rad_avg	, dfseg.curv_rad_min):#, dfseg.hotspot
        # if c==1:
            f.write('<Placemark>\n')
            f.write('\t<name>'+str(int(seg))+'</name>\n') #' '+str(int(a))+' '+str(int(b))+' '+
            f.write('\t<styleUrl>#marker_sty_red</styleUrl>\n') #marker_sty_'+str(i/10)+'
            f.write('\t<Point>\n')
            f.write('\t\t<coordinates>'+str(lo)+','+str(la)+',0</coordinates>\n')
            f.write('\t</Point>\n')
            f.write('</Placemark>\n')
    f.write('</Folder>\n')
