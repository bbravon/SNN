import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
los="1000 1001 1002 1003 1004 1005 1006 1100 1101 1102 1103 1104 1105 1106 1107 1108 1111 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2100 2101 2102 2103 2104 2200 2201 2202 2203 2204 2205 2206 2207 2208 2209 2300 2301 2400 2401 2402 2403 2404 2405 2406 2407 3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3100 3101 3102 3103 3104 3105 3106 3107 3108 3109 3110 3111 3112 3113 3114 3115 3116 3117 3118 3200 3201 3202 3203 3300 3301 3302 3303 3304 3305 3306 4000 4001 4002 4003 4004 4005 4006 4007 4100 4200 4201 4202 4203 5000"
codigos=los.split()
nuevoscodes=[]
for codigo in codigos:
    nuevoscodes.append(str(int(codigo)//100))
unicos=np.unique(nuevoscodes).tolist()
f = open("flotas.txt", "r")
names_flotas = []
for x in f:
    names_flotas.append(x.split())
from datetime import datetime, timedelta
import joblib
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib
rng = np.random.default_rng(12345)

def floter(escala="year",mes=None,dia=1,superstates=False,dinamico=True):
    all_data = []
    for j in [0,1,3,4]:
        if superstates==True:
            codigos=unicos
        else:
            codigos=los.split()
            
        vehicles=np.zeros((len(names_flotas[j]),len(codigos)),dtype=object)
        for i in range(len(names_flotas[j])):
    
            cat=pd.DataFrame(pd.read_csv("files/cats1/"+str(j)+"_"+str(i)+".csv"))
            
            if superstates==True:
                cat['CodigoRazon'] = cat['CodigoRazon'] // 100  # This already extracts the first two digits as integers
                cat['CodigoRazon'] = cat['CodigoRazon'].astype(str)  # Convert to string for the next step
                
            
            
            if escala!="year":
                
                cat["Fecha"] = pd.to_datetime(cat["Fecha"].astype(str), format='%y%m%d')
                start_date = datetime(2023, mes, dia) 
                if escala=="mes":
                    duracion=31
                if escala=="semana":
                    duracion=7
                if escala=="dia":
                    duracion=1
                end_date = start_date + timedelta(days=duracion) 
                cat = cat[(cat["Fecha"] >= start_date) & (cat["Fecha"] < end_date)].reset_index(drop=True)

            
            result = cat.groupby('CodigoRazon', as_index=False)['Duracion horas'].sum()
            result['CodigoRazon']=result['CodigoRazon'].astype(str)
            result["manada"]=j
           
            if dinamico==False:
                if superstates==False:
                    result=result[result['CodigoRazon']!="1000"]
                else:
                    result=result[result['CodigoRazon']!="10"]
            
                
            todosCode = pd.DataFrame({'CodigoRazon':codigos})
            df_merged = pd.merge(todosCode, result, on='CodigoRazon', how='left').fillna({'Duracion horas': 0})
            vehicles[i,:]=df_merged["Duracion horas"].values
            all_data.append([j] + list(vehicles[i, :]))
    
    column_names = ['Group'] + [f'codigo_{k}' for k in codigos]
    final_df = pd.DataFrame(all_data, columns=column_names)
    return final_df


def pdfer(df,name=None):
    pdfs=np.zeros(len(df),dtype=object)
    equipos=df.copy()
    equipos=equipos.drop(columns=["Group"])
    for i in range(len(equipos)):
        if equipos.loc[i].sum()!=0:
            
            pdfs[i]=np.array((equipos.loc[i]/equipos.loc[i].sum()).values)
        else:
            zeros=np.zeros(len(equipos.loc[i]))
            zeros[0]=1
            print("error",name,i)
            pdfs[i]=zeros
    return pdfs



def markover(flota,mes):
    # Generamos arrays para guardar los dataframe
    # Y los estados
    vehicleStates=np.zeros((len(names_flotas[flota])),dtype=object)

    # Registramos solo el mes
    
    # recorremos todos los equipos
    for k in range(len(names_flotas[flota])):
        
        # se crea el DataFrame de un solo vehiculo
        dato=pd.DataFrame(pd.read_csv("files/cats1/"+str(flota)+"_"+str(k)+".csv"))

        # Transformamos a datetime y filtramos para el mes en particular
        dato["Fecha"] = pd.to_datetime(dato["Fecha"].astype(str), format='%y%m%d')
        start_date = datetime(2023, mes, 1) 
        end_date = start_date + timedelta(days=30) 
        # Luego del filtro es importante reiniciar los índices para que sea accesible
        dato = dato[(dato["Fecha"] >= start_date) & (dato["Fecha"] < end_date)].reset_index(drop=True)

        
        statesMon=[]
        #Recorremos toda su trayectoria, viendo las entradas, salidas y tiempo en la entrada de los estados
        for i in range(len(dato)-1):
            state1=int(dato["CodigoRazon"].iloc[i])
            
            state2=int(dato["CodigoRazon"].iloc[i+1])
            state3=float(np.round(dato["Duracion horas"].loc[i],decimals=3))
            # Si el estado no se repite, se guarda. Esto es porque cualquier auto entrada es un error de input
            # En particular guardará el último. Considerar que quizás el tiempo en la entrada debería ser la suma y no solo el tiempo en el último.
            if state1!=state2:
                statesMon.append((state1,state2,state3))
        vehicleStates[k]=statesMon
    return vehicleStates




def markover2(flota,mes):
    # Generamos arrays para guardar los dataframe
    # Y los estados
    vehicleStates=np.zeros((len(names_flotas[flota])),dtype=object)

    # Registramos solo el mes
    
    # recorremos todos los equipos
    for k in range(len(names_flotas[flota])):
        
        # se crea el DataFrame de un solo vehiculo
        dato=pd.DataFrame(pd.read_csv("files/cats1/"+str(flota)+"_"+str(k)+".csv"))

        # Transformamos a datetime y filtramos para el mes en particular
        dato["Fecha"] = pd.to_datetime(dato["Fecha"].astype(str), format='%y%m%d')
        start_date = datetime(2023, mes, 1) 
        end_date = start_date + timedelta(days=30) 
        # Luego del filtro es importante reiniciar los índices para que sea accesible
        dato = dato[(dato["Fecha"] >= start_date) & (dato["Fecha"] < end_date)].reset_index(drop=True)

        
        statesMon=[]
        #Recorremos toda su trayectoria, viendo las entradas, salidas y tiempo en la entrada de los estados
        for i in range(len(dato)-1):
            state1=int(dato["CodigoRazon"].iloc[i])//100
            
            state2=int(dato["CodigoRazon"].iloc[i+1])//100
            state3=float(np.round(dato["Duracion horas"].loc[i],decimals=3))
            # Si el estado no se repite, se guarda. Esto es porque cualquier auto entrada es un error de input
            # En particular guardará el último. Considerar que quizás el tiempo en la entrada debería ser la suma y no solo el tiempo en el último.
            if state1!=state2:
                statesMon.append((state1,state2,state3))
        vehicleStates[k]=statesMon
    return vehicleStates


def markover3(flota,mes):
    # Generamos arrays para guardar los dataframe
    # Y los estados
    vehicleStates=np.zeros((len(names_flotas[flota])),dtype=object)

    # Registramos solo el mes
    
    # recorremos todos los equipos
    for k in range(len(names_flotas[flota])):
        
        # se crea el DataFrame de un solo vehiculo
        dato=pd.DataFrame(pd.read_csv("files/cats1/"+str(flota)+"_"+str(k)+".csv"))

        # Transformamos a datetime y filtramos para el mes en particular
        dato["Fecha"] = pd.to_datetime(dato["Fecha"].astype(str), format='%y%m%d')
        start_date = datetime(2023, mes, 1) 
        end_date = start_date + timedelta(days=30) 
        # Luego del filtro es importante reiniciar los índices para que sea accesible
        dato = dato[(dato["Fecha"] >= start_date) & (dato["Fecha"] < end_date)].reset_index(drop=True)

        
        statesMon=[]
        #Recorremos toda su trayectoria, viendo las entradas, salidas y tiempo en la entrada de los estados
        for i in range(len(dato)-1):
            state1=int(dato["CodigoRazon"].iloc[i])//100
            
            state2=int(dato["CodigoRazon"].iloc[i+1])//100
            state3=dato["Fecha"].iloc[i]
            # Si el estado no se repite, se guarda. Esto es porque cualquier auto entrada es un error de input
            # En particular guardará el último. Considerar que quizás el tiempo en la entrada debería ser la suma y no solo el tiempo en el último.
            if state1!=state2:
                statesMon.append((state1,state2,state3))
        vehicleStates[k]=statesMon
    return vehicleStates
    

def graph_builder(pares):
    G=nx.MultiDiGraph()
    for idx, edge in enumerate(pares, start=1):  # Start indexing at 1
        G.add_edge(edge[0], edge[1],weight=edge[2])
        G.nodes[edge[0]]['id'] = edge[0]  # Store node ID explicitly
        G.nodes[edge[1]]['id'] = edge[1]
    return G


def coloreo(nodo):
    code=int(str(nodo)[0])
    if code==1:
        return "skyblue"
    if code==2:
        return "teal"
    if code==3:
        return "orange"
    if code==4:
        return "lime"
    if code==5:
        return "silver"


def grafer2(grafo):
    G=grafo
    node_colors = [coloreo(node) for node in G.nodes()]
    # Use a layout for positioning nodes
    pos = nx.spring_layout(G)  # Ensure consistent layout
    
        # Start plotting
    plt.figure(figsize=(10, 8))
    
        # Draw the graph
    nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=700,
            node_color=node_colors,
            font_weight="bold",
            edge_color="gray",
            arrows=True,  # Show arrows only for directed graphs
            arrowstyle="->",
            arrowsize=15)
    
    plt.title("Trayectoria")
    plt.show()


def costeo(nodo):
    code=int(str(nodo.get("id"))[0])
    if code==1:
        return 1
    if code==2:
        return 3
    if code==3:
        return 10
    if code==4:
        return 10
    if code==5:
        return 10




def edge_subst_cost(edge1, edge2):
    # If both edges have the 'weight' attribute, calculate the cost as the absolute difference
    if 'weight' in edge1 and 'weight' in edge2:
        return abs(edge1['weight'] - edge2['weight'])
    # Otherwise, return a default substitution cost
    return 1

# Custom edge deletion cost function
def edge_del_cost(edge):
    # Return a fixed cost for edge deletion (can be customized)
    return abs(edge["weight"])

# Custom edge insertion cost function
def edge_ins_cost(edge):
    # Return a fixed cost for edge insertion (can be customized)
    return abs(edge["weight"])


def node_subst_cost(node1, node2):
    return abs(costeo(node1)-costeo(node2))

def node_del_cost(node):
    return abs(costeo(node))

def node_ins_cost(node):
    return abs(costeo(node))

def compute_chunk_geds(chunk_ids, map_graphs,node=False,edge=True):
    results = []
    for ix1, ix2 in chunk_ids:
        g1 = map_graphs[ix1]
        g2 = map_graphs[ix2]
        if node==False and edge==True:
            ged = nx.graph_edit_distance(g1,g2,edge_subst_cost=edge_subst_cost,edge_ins_cost=edge_ins_cost, edge_del_cost=edge_del_cost,timeout=1)
        if node==True and edge==True:
            ged = nx.graph_edit_distance(g1,g2,node_subst_cost=node_subst_cost,node_del_cost=node_del_cost, node_ins_cost=node_ins_cost,edge_subst_cost=edge_subst_cost,edge_ins_cost=edge_ins_cost, edge_del_cost=edge_del_cost,timeout=1)
        if node==True and edge==False:
            ged = nx.graph_edit_distance(g1,g2,node_subst_cost=node_subst_cost,node_del_cost=node_del_cost, node_ins_cost=node_ins_cost,timeout=1)
        if node==False and edge==False:
            ged = nx.graph_edit_distance(g1,g2,timeout=1)
        results.append((ix1, ix2, ged))
    return results


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()



def paralelizador_ged(list_of_graphs,chunk_size=8,node=False,edge=True):
    num_graphs = len(list_of_graphs)
    map_graphs = {idx: g for idx, g in enumerate(list_of_graphs)}
    
    ged_matrix = np.full((num_graphs, num_graphs), np.inf)  # Use `np.inf` as a placeholder
 
    pairs_graph_ids = list(itertools.combinations(range(num_graphs), r=2))
    chunk_into_chunks = lambda ll, n: [ll[i * n:(i + 1) * n] for i in range((len(ll) + n - 1) // n)]
    chunksof_pairs_graph_ids = chunk_into_chunks(pairs_graph_ids, chunk_size)
    
    n_jobs = -1  # Use all available cores
    
    with tqdm_joblib(tqdm(desc="Calculating GEDs", total=len(chunksof_pairs_graph_ids))) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_chunk_geds)(chunk_pairs, map_graphs,node=node,edge=edge)
            for chunk_pairs in chunksof_pairs_graph_ids
        )

    for chunk in results:
        for ix1, ix2, ged in chunk:
            ged_matrix[ix1, ix2] = ged
            ged_matrix[ix2, ix1] = ged  # Use symmetry
    
    # Replace `np.inf` with 0 for diagonal elements
    np.fill_diagonal(ged_matrix, 0)
    return ged_matrix



# Dada una inicial posición, una trayectoria y una cantidad n de horas, devuelve la posición donde se ubicaba 
# la trayectoria n horas antes de la posición inicial.

def time_travel(position,trayectoria,horas):
    cumsum=np.cumsum(np.array(trayectoria)[:,2])
    target=abs(cumsum[position]-horas) # Tiene un valor absoluto por si se pasa
    
    closest_index = np.argmin(np.abs(cumsum - target)) 
    return closest_index

def crononauta(positions,hora,trayectoria,gap=0,add=0):
    trayectos=[]

    # Vamos hacia adelante. previous = 0 es la primera longitud
    previous=0
    trayecto=[]

    # por cada posición
    for pos in positions:

        # Obtenemos el indice de n horas antes con time_travel:
        index=time_travel(pos,trayectoria,hora)
        pos2=time_travel(pos,trayectoria,gap)
        # Si el índice obtenido es antes del último registro
        if index<previous:
            #print(f"solapamiento con {index,pos}")
            # Avisar que existe un solapamiento
            #trayecto=trayectoria[previous:pos]

            # No añadimos ese trayecto
            trayecto=[]
        else:

            # Si no hay solapamiento, lo añadimos. Notar que habrá solapamiento hasta que un index esté por encima o igual a un previous
            trayecto=trayectoria[index:pos2+add]
        previous=pos # Se actualiza el previo
        
        trayectos.append(trayecto) # guardamos el trayecto
    return trayectos

def analisis(flota,mes,horas,reparacion=[30],superstate=False,gap=0):
    
    if superstate==False:
        s=100
        cadenas=markover(flota,mes)
    else:
        cadenas=markover2(flota,mes)
        s=1
    trayectos_cadenas=[]
    # por cada equipo en la cadena
    for equipo in cadenas:
        # obtenemos donde ocurre el estado de reparacion
        if len(equipo)>0:
            
            positions=np.where(np.isin(np.array(equipo)[:,1]//s, reparacion))[0]
            
            trayectos_equipo=np.array(crononauta(positions,horas,equipo,gap=gap),dtype=object)
            len_array = np.array(list(map(len, trayectos_equipo)))
            mask=np.where(len_array!=0)
            trayectos_cadenas.append(trayectos_equipo[mask])
        else:
            trayectos_cadenas.append([])
    return np.array(trayectos_cadenas,dtype=object)

def analisis_rand(flota,mes,horas,reparacion=[30],superstate=False,gap=0):
    
    if superstate==False:
        s=100
        cadenas=markover(flota,mes)
    else:
        cadenas=markover2(flota,mes)
        s=1
    trayectos_cadenas=[]
    # por cada equipo en la cadena
    for equipo in cadenas:
        # obtenemos donde ocurre el estado de reparacion
        if len(equipo)>20:
            positions=rng.uniform(low=10, high=len(equipo), size=7).astype(int)
            trayectos_equipo=np.array(crononauta(positions,horas,equipo,gap=gap),dtype=object)
            len_array = np.array(list(map(len, trayectos_equipo)))
            mask=np.where(len_array!=0)
            trayectos_cadenas.append(trayectos_equipo[mask])
        else:
            trayectos_cadenas.append([])
    return np.array(trayectos_cadenas,dtype=object)




#### GRAKELS ####

from grakel import Graph, Kernel

def convert_multigraph_to_grakel2(multigraph):
    """
    Convert a NetworkX multigraph to a GraKeL-compatible format with vertex labels.
    - Aggregates edge weights.
    - Assigns default or existing vertex labels.
    """
    # Create a simple graph by aggregating edge weights
    simple_graph = nx.Graph()
    for u, v, data in multigraph.edges(data=True):
        weight = data.get('weight', 1)  # Default weight is 1
        if simple_graph.has_edge(u, v):
            simple_graph[u][v]['weight'] += weight  # Aggregate weights
        else:
            simple_graph.add_edge(u, v, weight=weight)
    
    # Assign default labels if none exist
    
    labels = {node: f"label_{node}" for node in multigraph.nodes()}
    #labels = {node: data.get('label', f"label_{node}") for node, data in simple_graph.nodes(data=True)}

    edge_list = list(simple_graph.edges())
    edge_labels = {(u, v): simple_graph.get_edge_data(u, v) for u, v in edge_list}
    # Convert to GraKeL format
    grakel_graph = Graph(
        edge_list,
        edge_labels=edge_labels,
        node_labels=labels
    )
    return grakel_graph

def grakelizador(lista_grafos):
    grak=[]
    for i, hipergrafo2 in enumerate(lista_grafos):
        try:
            grak.append(convert_multigraph_to_grakel2(hipergrafo2))
        except ValueError as e:
            print(f"Error converting graph at {lista_grafos}[{i}]: {e}")
    return grak


from grakel import RandomWalkLabeled
def distance_grakel(grak1, grak2,n):
    wl_kernel = RandomWalkLabeled(n_jobs=n, normalize=True)
    grakel_graphs= grak1 + grak2
    kernel_matrix = wl_kernel.fit_transform(grakel_graphs)
    n = kernel_matrix.shape[0]
    distance_matrix = np.sqrt(
      np.add.outer(np.diag(kernel_matrix), np.diag(kernel_matrix)) - 2 * kernel_matrix)
    

    
    return distance_matrix


def distanciasWL(flota,mes,horas,reparacion=30,gap=0):
    analisis1=analisis(flota,mes,horas,reparacion,gap=gap)
    graphs=[]
    graphs_per_group=[]
    for group in analisis1:
        grafos_group=[]
        for chain in group:
            grafos_group.append(graph_builder(chain))
        graphs_per_group.append(len(grafos_group))
        graphs.append(grafos_group)
    list_of_graphs=[]
    longitudes=[]
    for group in graphs:
        if len(group)!=0:
            list_of_graphs=list_of_graphs+group
            longitudes.append(len(group))
    graks=grakelizador(list_of_graphs)
    ged=distance_grakel(graks,[],3)
    return ged,longitudes

def distanciasWL_rand(flota,mes,horas,reparacion=30,gap=0):
    analisis1=analisis_rand(flota,mes,horas,reparacion,gap=gap)
    graphs=[]
    graphs_per_group=[]
    for group in analisis1:
        grafos_group=[]
        for chain in group:
            grafos_group.append(graph_builder(chain))
        graphs_per_group.append(len(grafos_group))
        graphs.append(grafos_group)
    list_of_graphs=[]
    longitudes=[]
    for group in graphs:
        if len(group)!=0:
            list_of_graphs=list_of_graphs+group
            longitudes.append(len(group))
    graks=grakelizador(list_of_graphs)
    ged=distance_grakel(graks,[],10)
    return ged,longitudes
    

def dater_randWL(mes,reparacion,rangos):
    n=len(rangos)
    info=np.zeros(4,dtype=object)
    info0=np.zeros((n,2),dtype=object)
    info1=np.zeros((n,2),dtype=object)
    info3=np.zeros((n,2),dtype=object)
    info4=np.zeros((n,2),dtype=object)
    for i in range(n):
        info0[i,:]=distanciasWL_rand(0,mes,rangos[i],reparacion=reparacion)
        info1[i,:]=distanciasWL_rand(1,mes,rangos[i],reparacion=reparacion)
        info3[i,:]=distanciasWL_rand(3,mes,rangos[i],reparacion=reparacion)
        info4[i,:]=distanciasWL_rand(4,mes,rangos[i],reparacion=reparacion)
    info[0]=info0
    info[1]=info1
    info[2]=info3
    info[3]=info4
    return info

