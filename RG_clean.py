import json
import numpy as np
import pandas as pd
import math
from collections import Counter


rad_earth_m = 6373000 # Radius of the earth in meters
metros_acceso_rotonda=100 #los 100 ultimos metros antes de lllegar a la roton son roton
MAX_DIST_CRASH_ROAD = 200 # A MAS DE MAX_DIST_CRASH_ROAD metros el crash no está en la road
weaving_segment_len=800
curv_radius_max = 10000
dfpoints_path = "../data/pts.pkl"
hora_bins=8 # num bins for discretization "hora" var
speed_bins=6 # num bins for discretization "speed_limit" and "speed_limit_adj" var
radius_discret_edges = [-1, -0.001, 50, 100, 200, 600, float('inf')]
radius_discret_labels = [5,4,3,2,1,0]




def join_ways(dfways, segments):
    # remove duplicates from dfways and joined adjacent segments
    # print("void duplicates?: " , all((dfways.loc[dfways.duplicated('way_id')]).drop(['pt_end','pt_ini','way_id'], axis=1).isnull() ) )
    if len(dfways.loc[dfways.duplicated('way_id')].drop(['pt_end','pt_ini','way_id'], axis=1)):
        print("join_ways::warning, there are duplicate ways")
    # print("lloking for nd=1367734467. aparece en rectas en dos cabezas per no se junta, xq??")
    # 'pt_end','pt_ini','way_id'
    # dfways=dfways.drop_duplicates('way_id', 'first')
    p=np.unique(dfways['pt_ini'].tolist() + dfways['pt_end'].dropna().tolist() )
    joined=set([w for w in dfways.way_id]) #
    rectas = []

    for i in p:
        res=dfways.loc[(dfways['pt_end']==i) | (dfways['pt_ini']==i), 'way_id']
        if len(res)==2:
            # join the ways with id res.iloc[0] and res.iloc[1]
            df1, df2 = segments[res.iloc[0]], segments[res.iloc[1]]
            if df1.p_id.iloc[-1] != df1.p_id.iloc[0] and df2.p_id.iloc[-1] != df2.p_id.iloc[0]:
                # if first point == last point and len(res)==2: se va a unir consigo misma
                if df1.p_id.iloc[-1] == df2.p_id.iloc[0]:
                    df=pd.concat([df1[:-1], df2], ignore_index=True)
                elif df1.p_id.iloc[-1] == df2.p_id.iloc[-1]:
                    df2=df2.sort_index(ascending=False)
                    df=pd.concat([df1[:-1], df2], ignore_index=True)
                elif df2.p_id.iloc[-1] == df1.p_id.iloc[0]:
                    df=pd.concat([df2[:-1], df1], ignore_index=True)
                elif df2.p_id.iloc[0] == df1.p_id.iloc[0]:
                    df1=df1.sort_index(ascending=False)
                    df=pd.concat([df1[:-1], df2], ignore_index=True)
                else:
                    print("intersection type T to the middle of the way")
                    print (dfways.loc[(dfways['pt_end']==i) | (dfways['pt_ini']==i), ['way_id', 'pt_end', 'pt_ini']])
                    print(df2.p_id.tolist(), df1.p_id.tolist())
                    print(df2.way_id.tolist(), df1.way_id.tolist())
                ways = np.unique(df.way_id)
                for w in ways:
                    segments[w]=df #todos los way_id apuntan al nuevo df concatenado
                joined=joined.difference(ways)
                joined.add(max(ways))
    for w_id in joined:
        rectas.append(segments[w_id])
    return dfways, rectas


def distance_on_unit_sphere(lat1, long1, lat2, long2):
    # From https://github.com/adamfranco/curvature
    # From http://www.johndcook.com/python_longitude_latitude.html
    if lat1 == lat2	 and long1 == long2:
        return 0
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
    # print(phi1, phi2, theta1, theta2, cos, arc)
    return arc



def segments2coords(rectas, dfcoord, dfways):
    rectas2=[]
    for df in rectas:
        df2 = pd.merge(df, dfcoord, how='left', left_on='p_id', right_on='id').reset_index(drop=True)
        df3 = pd.DataFrame(columns=['ini_pt', 'end_pt', 'ini_lat', 'ini_lon', 'end_lat', 'end_lon', 'way_id', 'color', 'junction','num_crash','destination_road'])
        df3[['way_id', 'ini_lat', 'ini_lon', 'ini_pt']]=df2[['way_id', 'lat', 'lon', 'p_id']]
        df3['ini_lat'] = df3['ini_lat'].astype(float)
        df3['ini_lon'] = df3['ini_lon'].astype(float)
        df3[['end_lat', 'end_lon', 'end_pt']] = df3[['ini_lat', 'ini_lon', 'ini_pt']].shift(-1)
        rectas2.append(df3)# last line has null ['end_lat', 'end_lon', 'end_pt']
    return rectas2


def label_manually(dfways):
    print("If any way in OSM is incorrectly labelled you can label it here.")
    # dfways.loc[dfways.way_id=='50991051','ref']='N-634'
    # dfways.loc[dfways.way_id=='14195025','ref']='N-634'
    # dfways.loc[dfways.way_id=='14195025','highway']='trunk'
    # dfways.loc[dfways.way_id=='50991042','highway']='trunk'
    # dfways.loc[dfways.way_id=='50991042','ref']='N-634'
    # dfways.loc[dfways.way_id=='35603988', 'ref']='N-634'
    # dfways.loc[dfways.ref=='AP-1;AP-8', 'ref']='AP-8'
    # dfways.loc[dfways.way_id=='198038777','highway']='trunk'
    # # desdobles ..
    # aux = ['4486010','27062497','333355923','333355918','441624487','27080428',
    #     '27080430','27080780','27080781','238783151','377094546', '10276051',
    #     '217085417', '193674416', '44580274','378363382','203299935','378363385',
    #     '238784480']
    # dfways.loc[dfways.way_id.isin(aux),'ref']=''

def set_junctions_with_residentials(dfcoord, rectas, residential_map):
    with open(residential_map) as data_file:
        data = json.load(data_file)
    coords = []
    segments = {}
    tagslist = []
    for place in data['elements']:
        if place['type']=='node':
            coords.append(place['id'])
    inters = np.intersect1d(coords, dfcoord.id.values)
    for df in rectas:
        df['junction']=None
        df.loc[df.ini_pt.isin(inters), 'junction']='intersect' #df.ini_pt.isin(multi_pt) &
        df.loc[df.end_pt.isin(inters), 'junction']='intersect' #& df.junction.isnull()
    return inters

def set_junctions(dfways, rectas):
    path_head={df.ini_pt.iloc[0] for df in rectas} | {df.ini_pt.iloc[-1] for df in rectas}
    link =dfways.loc[dfways.highway.str.endswith('_link'), 'way_id']
    roton=dfways.loc[dfways.junction=='roundabout', 'way_id']
    return [set_junctions_aux(df, link, roton, path_head) for df in rectas]

def set_junctions_aux(df, link, roton, path_head):
    df['length1'] = [distance_on_unit_sphere(la, lo, lat, lon) * rad_earth_m for la, lo, lat, lon in zip(df.ini_lat, df.ini_lon, df.end_lat, df.end_lon)]
    # df['junction']=None en set_junctions_with_residentials
    df.loc[df.way_id.isin(link),'junction']  ='link'
    df.loc[df.way_id.isin(roton),'junction']='roundabout'
    df.loc[df.ini_pt.isin(path_head) & df.junction.isnull(), 'junction']='intersect' #df.ini_pt.isin(multi_pt) &
    df.loc[df.end_pt.isin(path_head) & df.junction.isnull(), 'junction']='intersect' #& df.junction.isnull()
    df['junction']=df.junction.fillna('nojunct')
    # set the shor 'rectas' between 'intersect' as 'intersect'
    df['block'] = (df.junction.shift(1) != df.junction).astype(int).cumsum()
    aux=df.groupby(['junction','block']).sum().length1.reset_index(name='len').sort_values(by='block').reset_index(drop=True)
    aux['weaving_segment']= (aux.junction=='nojunct') & (aux.len<weaving_segment_len) & (aux.shift(-1).junction=='intersect' )& (aux.shift(1).junction=='intersect' )
    # ojo copia
    df2=pd.merge(df, aux, on=['junction','block'])
    df2.loc[df2.weaving_segment, 'junction']='intersect'
    df2=df2.drop(['weaving_segment', 'block','len'], axis=1)
    return df2


def check_last_line(rectas):#previously known as correct_last_line
    rectas2=[]
    for df in rectas:
        indnull = df[df['end_pt'].isnull()].index
        if len(indnull)==1 and indnull[0]==len(df)-1:
            df = df[:-1]
        elif (len(indnull)==1 and indnull[0]!=len(df)-1) or len(indnull)>1:
            print(len(indnull)==1 , indnull[0]!=len(df)-1 , len(indnull)>1, indnull)
            print(df)
            raise Exception("ERROR: There is a null row in the middle of the path")
        rectas2.append(df)
    return rectas2



def select_roads(dfways, rectas, roads):
    # dfways_small=dfways.loc[dfways['ref'].isin(dfcrash.Carretera.unique())].copy()
    dfways_small = dfways.loc[dfways['highway'].isin(roads)].copy()
    # print("len(dfways),len(dfways_small)",len(dfways),len(dfways_small) )
    print("Discarding ways: from",len(dfways),"to",len(dfways_small))
    rectas_small = [df for df in rectas if len(np.intersect1d(df.way_id.unique(), dfways_small.way_id.unique()))!=0]
    rec=[]
    for df in rectas_small:
        df['ind']=list(range(len(df)))
        aux=pd.merge(df,dfways_small[['way_id','ref']])
        aux.columns=[i if i!='ref' else 'road' for i in aux.columns]
        # check that we have not deleted a segment in the middle of the recta
        if max(aux.ind) - min(aux.ind) + 1 != len(aux):
            print("Warning: we have deleted a segment in the middle. Roads:", aux.road.unique())
            print("Cause: the current path is labelled with different road types in OSM")
        rec.append(aux.drop('ind', axis=1))
    rectas_small=rec
    return dfways_small, rectas_small


def join_reduced_ways(rectas):
    count=[]
    for df in rectas:
        if all(df.junction!='roundabout') :#and len(df)>1
            count.append(df.ini_pt.iloc[0])
            count.append(df.end_pt.iloc[-1])
    count = Counter(count)
    pts = [key for key in count.keys() if count[key]==2]
    # print(len(pts))
    for pt in pts:
        #check_index(rectas)
        ds=get_df2_ending_pt(rectas, pt)
        if len(ds)==2 and ds[0]!=ds[1]: # check d1!=d2 and assume enc[0]<enc[1]
            df2 = rectas.pop(ds[1])
            df1 = rectas.pop(ds[0])
            if df1.end_pt.iloc[-1] == df2.ini_pt.iloc[0]:
                df=pd.concat([df1, df2], ignore_index=True)
            elif df2.end_pt.iloc[-1] == df1.ini_pt.iloc[0]:
                df=pd.concat([df2, df1], ignore_index=True)
            elif df1.end_pt.iloc[-1] == df2.end_pt.iloc[-1]:
                X=df2[::-1]
                X[['ini_pt','end_pt','ini_lat', 'ini_lon', 'end_lat', 'end_lon']]=X[['end_pt','ini_pt','end_lat', 'end_lon', 'ini_lat', 'ini_lon']]
                df=df1.append(X, ignore_index=True)
            elif df2.ini_pt.iloc[0] == df1.ini_pt.iloc[0]:
                X=df2[::-1]
                X[['ini_pt','end_pt','ini_lat', 'ini_lon', 'end_lat', 'end_lon']]=X[['end_pt','ini_pt','end_lat', 'end_lon', 'ini_lat', 'ini_lon']]
                df=X.append(df1, ignore_index=True)
            else:
                print("Something wrong in join_reduced_ways")
            rectas.append(df)
        else:
            print("Warning. A point appearns twice but posibly in the same path. Probable cause: a roundabout is uncorrectly labelled in OSM. Point",pt)

def get_df2_ending_pt(rectas, pt):
    # lis of dfs 'rectas', return indices of both begining or ending at pt
    # pt appears twice, check becaues maybe is the begining AND ending of the same df
    # enc[0]<enc[1]
    i=0
    found=[]
    while len(found)<2 and i<len(rectas):
        a=rectas[i].ini_pt.iloc[0]
        b=rectas[i].end_pt.iloc[-1]
        if a==b and a==i:
            return [i,i] # roundabout?
        if a==pt or b==pt:
            found.append(i)
        i+=1
    return found


def curvature(rectas):
    # adapted from https://github.com/adamfranco/curvature
    for df in rectas:
        #curv_radius, slope
        df['length1'] = [distance_on_unit_sphere(la, lo, lat, lon) * rad_earth_m for la, lo, lat, lon in zip(df.ini_lat, df.ini_lon, df.end_lat, df.end_lon)]
        if df.end_pt.iloc[-1]==df.ini_pt.iloc[0]: #roton
            mid = df.iloc[int(len(df)/2)] #the one right in fron on the roundabout
            la, lo = df.ini_lat.ix[0],df.ini_lon.ix[0]
            lat,lon = mid.ini_lat,mid.ini_lon
            dis = distance_on_unit_sphere(la, lo, lat, lon) * rad_earth_m
            df['curv_radius']=int(dis/2)
            # df['junction']='roundabout'
            # print("roton todas las curvatures = ",df.curv_radius.unique())
        else:
            df['length2'] = [distance_on_unit_sphere(la, lo, lat, lon) * rad_earth_m for la, lo, lat, lon in zip(df.ini_lat, df.ini_lon, df.end_lat.shift(-1), df.end_lon.shift(-1))]
            df['length3'] = [distance_on_unit_sphere(la, lo, lat, lon) * rad_earth_m for la, lo, lat, lon in zip(df.ini_lat, df.ini_lon, df.end_lat.shift(-2), df.end_lon.shift(-2))]
            df['length4'] = [distance_on_unit_sphere(la, lo, lat, lon) * rad_earth_m for la, lo, lat, lon in zip(df.ini_lat, df.ini_lon, df.end_lat.shift(-3), df.end_lon.shift(-3))]
            df['length6'] = [distance_on_unit_sphere(la, lo, lat, lon) * rad_earth_m for la, lo, lat, lon in zip(df.ini_lat, df.ini_lon, df.end_lat.shift(-5), df.end_lon.shift(-5))]
            df['curv_radius']  = [(a * b * c)/math.sqrt(math.fabs((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))) for a, b, c in zip(df.length1, df.length1.shift(-1), df.length2)]
            df['curv_radius2'] = [(a * b * c)/math.sqrt(math.fabs((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))) for a, b, c in zip(df.length2, df.length2.shift(-2), df.length4)]
            df['curv_radius3'] = [(a * b * c)/math.sqrt(math.fabs((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))) for a, b, c in zip(df.length3, df.length3.shift(-3), df.length6)]
            if len(df) > 1:
                df.ix[len(df)-1, 'curv_radius'] = df.ix[len(df)-2, 'curv_radius']
            df.loc[df.curv_radius3>curv_radius_max,'curv_radius3']=curv_radius_max
            df.loc[df.curv_radius2>curv_radius_max,'curv_radius2']=curv_radius_max
            df.loc[df.curv_radius >curv_radius_max,'curv_radius'] =curv_radius_max
            # **Curvature correction by smoothing the lines**
            df['curv_radius'] = (
            #    0.1*df['curv_radius'] + 0.1*df['curv_radius'].shift()
                + 0.15*df['curv_radius2'].shift(2) + 0.15*df['curv_radius2'].shift(1)
                + 0.7*df['curv_radius3'].shift(3)
                ).fillna(df['curv_radius'])
            if len(df) == 1:
                df.ix[0,'curv_radius']=curv_radius_max
            df.drop(['length2', 'length3','length4','length6','curv_radius2','curv_radius3'], axis=1, inplace=True)



def segment_roundabouts(rectas):
        #rotondas y adyacentes => seg_ig
        # si un df in rectas tiene algun cacho como roton =>seg_id
        # si un df es adjacente a una rotos => los 'metros_acceso_rotonda' se ponen en el mismo seg_id
    dfcoords_roundabout={}
    rec=[]
    contsegment=0
    dfround = pd.DataFrame(columns=['pt_id', 'seg_id'])
    for df in rectas:
        df['seg_id']=None #INI df['seg_id']=None
        if any(df.junction=='roundabout'):
            df.seg_id=contsegment
            aux = pd.DataFrame(columns=['pt_id', 'seg_id'])
            aux['pt_id']= df.ini_pt.unique() #+ [df.end_pt.iloc[-1]]
            aux['seg_id']=contsegment
            dfround=pd.concat([dfround, aux])
            contsegment+=1
    # print("numero roton encontrada",contsegment)
    for df in rectas:
        if 'roundabout' not in df.junction.unique() and any(dfround.pt_id== df.ini_pt.ix[0]):
            df.loc[df.length1.cumsum()<=metros_acceso_rotonda, 'seg_id']=dfround.loc[dfround.pt_id== df.ini_pt.ix[0],'seg_id'].values[0]
        if 'roundabout' not in df.junction.unique() and any(dfround.pt_id==df.end_pt.iloc[-1]):
            cum2=df.length1[::-1].cumsum()[::-1]
            df.loc[cum2<=metros_acceso_rotonda, 'seg_id']=dfround.loc[dfround.pt_id== df.end_pt.iloc[-1],'seg_id'].values[0]
    return contsegment

def segments_tunnels(rectas,contsegment, dfways): #tunnels=> seg_id
    ways=dfways.loc[dfways.tunnel=='yes', 'way_id']
    for df in rectas:
        df['tunnel']=False
        if any(df.way_id.isin(ways)):
            df.loc[df.way_id.isin(ways), 'tunnel']=True
            df['block'] = (df.tunnel.shift(1) != df.tunnel).astype(int).cumsum() #starts by 1
            aux = df.groupby(['block', 'tunnel']).size().reset_index(name='len').sort_values(by='block').reset_index(drop=True)
            aux = aux[aux.tunnel]
            aux['seg']=[contsegment+i for i in range(len(aux))]
            contsegment += len(aux)
            df2 = pd.merge(df, aux[['block', 'seg']], how='left')
            df.seg_id.fillna(value=df2.seg, inplace=True)
            df.drop('block', axis=1,inplace=True)
            # print("contunnel",contsegment)
    return contsegment

def segment_weaving(rectas,contsegment): #weaving segments => seg_ig
    rectas2=[]
    for df in rectas:
        aux = (df.junction.shift(1) != df.junction ) | ( df.tunnel.shift(1) != df.tunnel)
        df['block'] = aux.astype(int).cumsum() #starts by 1
        aux = df.loc[(df.seg_id.isnull()) & (df.junction=='intersect')]
        if len(aux)>1:#the last one
            aux = aux.groupby(['block', 'junction']).size().reset_index(name='len').sort_values(by='block').reset_index(drop=True)
            aux['seg'] = [contsegment+i for i in range(len(aux))]
            contsegment += len(aux)
            df2 = pd.merge(df, aux[['block', 'seg']], how='left')
            df.seg_id.fillna(value=df2.seg, inplace=True)
        df.drop('block', axis=1,inplace=True)
    return contsegment

def segment_curves(rectas,contsegment):
    rectas2=[]
    for df in rectas:
        if any(df.seg_id.isnull()):
            df['color'] = pd.cut(df.curv_radius, bins=radius_discret_edges, labels=radius_discret_labels)
            df.loc[df.color!=0,'color']=1 ##solo interesa separar las rectas de las curvas, no su angulo
            aux = (df.junction.shift(1) != df.junction ) | ( df.tunnel.shift(1) != df.tunnel) | (df.color.shift(1) != df.color ) | ( df.seg_id.notnull())| ( df.shift().seg_id.notnull())
            df['block'] = aux.astype(int).cumsum() #starts by 1
            aux=df[df.seg_id.isnull()]
            aux=aux.groupby(['block']).size().reset_index(name='len').sort_values(by='block').reset_index(drop=True)
            aux['seg']=[contsegment+i for i in range(len(aux))]
            contsegment+=len(aux)
            df2=pd.merge(df, aux[['block', 'seg']], how='left')
            df2.seg_id=df2.seg_id.fillna(value=df2.seg)
            df2=df2.drop(['seg', 'block'], axis=1)
            rectas2.append(df2)
        else:
            rectas2.append(df)
    return rectas2,contsegment


def segment_summarize(dfrectas):
    # dfrectas['seg_id']=dfrectas['seg_id'].astype(int)
    # OJO if a segment has no adjacent it is droped in the meatge
    # de los segmentos con rotonda pasamos del cacho de las incorporaciones
    roton=dfrectas.loc[dfrectas.junction=='roundabout','seg_id'].unique()
    dfrectas.loc[(dfrectas.seg_id.isin(roton)) & (dfrectas.junction!='roundabout'),'curv_radius']=None

    # a segment roundabout will have some meters of roundabout and some of intersection
    # a segment which is not a roundabout will be either intersection XOR nojunct
    junct_in_seg = dfrectas.groupby(['seg_id'])['junction'].unique().apply(tuple).reset_index(name='lista')#['intersect', 'roundabout', 'nojunct'],
    junct_in_seg.groupby('lista').size().reset_index(name='coun').columns
    gr=dfrectas.groupby(['seg_id', 'junction']) #, as_index=False
    df_junction_meters=gr.length1.sum().unstack().fillna(0)
    df_junction_meters=df_junction_meters.reset_index()
    gr=dfrectas.groupby(['seg_id'] )#, as_index=False)
    df=gr.agg({        #'junction' : lambda x: tuple(np.sort(x.unique())),
            'curv_radius': {
                'curv_rad_avg' : np.sum,
                'curv_rad_min' : np.min,
                'count_lines' : 'count'
            },
            'length1' : {
                'length' : np.sum
            },
            'tunnel' : {
                'tunnel' : lambda x: list(x.unique())[0]
            },
             'road': {
                 'road': lambda x: Counter(x).most_common(1)[0][0],
                 'road_list': lambda x: tuple(x.unique()),
                 'road_count': lambda x: len(tuple(x.unique()))
             }
           }).reset_index()
    df.columns=df.columns.droplevel() #.reset_index()
    df.columns=[name if name!='' else 'seg_id' for name in df.columns]
    dfseg = pd.merge(df, df_junction_meters)
    if 'roundabout' not in dfseg.columns:
        dfseg['roundabout'] = 0
    dfseg[['intersect', 'nojunct', 'roundabout']] = dfseg[['intersect','nojunct'	,'roundabout']].div(df.length.astype(float), axis='index')
    dfseg['curv_rad_avg']=dfseg['curv_rad_avg']/dfseg['count_lines']
    # OJO binary geometry
    dfseg = dfseg.drop('nojunct', axis=1)
    dfseg.loc[dfseg.roundabout!=0,'roundabout']=1
    dfseg.loc[dfseg.roundabout!=0,'intersect']=1
    dfseg.loc[dfseg.intersect!=0,'intersect']=1
    dfseg.roundabout = dfseg.roundabout.astype(bool)
    dfseg.intersect = dfseg.intersect.astype(bool)
    # find the adjacent segments
    pt_seg=pd.concat([dfrectas[['ini_pt','seg_id']], dfrectas[['end_pt','seg_id']] ])
    pt_seg['ini_pt']=pt_seg['ini_pt'].fillna(value=pt_seg.end_pt)
    pt_seg = pt_seg.drop('end_pt', axis=1)
    pt_seg.columns = [i if i!='ini_pt' else 'pt_id' for i in pt_seg.columns]
    pt_seg = pt_seg.drop_duplicates()
    pt_seg = pt_seg.dropna()
    aux = pt_seg.groupby('pt_id').size().reset_index(name='repes')
    aux = aux[aux.repes>1]
    pt_seg = pd.merge(pt_seg, aux, on='pt_id').drop('repes', axis=1)
    adjacency = pd.merge(pt_seg, pt_seg, right_on='pt_id', left_on='pt_id')
    adjacency = adjacency[adjacency.seg_id_x != adjacency.seg_id_y].drop('pt_id', axis=1)
    # [['seg_id_x', 'length', 'curv_rad_min','curv_rad_avg','intersect','nojunct','roundabout', 'speed_limit']]
    seg_adj = pd.merge(adjacency ,dfseg , left_on='seg_id_y', right_on='seg_id')[['seg_id_x', 'length', 'curv_rad_min','curv_rad_avg','intersect','roundabout']]
    # ['seg_id_x', 'length', 'curv_rad_min','curv_rad_avg','intersect','roundabout', 'speed_limit']]
    gr=seg_adj.groupby(['seg_id_x'] )#, as_index=False)
    seg_adj = gr.agg({
            'length': {
                'length_adj' : np.sum,
            },
            'curv_rad_min' : {
                'curv_adj_min' : np.min,
                # 'curv_adj_max' : np.max,
            },
            'curv_rad_avg' : {
                'curv_adj_mean' : np.mean,
            },
            'intersect': {
                'intersect_adj': np.any,
            },
            'roundabout': {
                'roundabout_adj': np.any,
            },
           }).reset_index(drop=False)
    seg_adj.columns = [j if j!='' else 'seg_id'  for i,j in zip(seg_adj.columns.get_level_values(0), seg_adj.columns.get_level_values(1))]
    dfseg = pd.merge(dfseg, seg_adj)
    return dfseg, adjacency















# end    #
