# Funciones 
# Pipeline: CSV -> markover(csv)-> sampler_regressor(markover)-> sequence_generator(Sampler_regressor) -> Modelo.fit(sequence_generator)






import scipy 
import datetime
import numpy as np
import pandas as pd

los="1000 1001 1002 1003 1004 1005 1006 1100 1101 1102 1103 1104 1105 1106 1107 1108 1111 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2100 2101 2102 2103 2104 2200 2201 2202 2203 2204 2205 2206 2207 2208 2209 2300 2301 2400 2401 2402 2403 2404 2405 2406 2407 3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3100 3101 3102 3103 3104 3105 3106 3107 3108 3109 3110 3111 3112 3113 3114 3115 3116 3117 3118 3200 3201 3202 3203 3300 3301 3302 3303 3304 3305 3306 4000 4001 4002 4003 4004 4005 4006 4007 4100 4200 4201 4202 4203 5000"
codigos=los.split()
nuevoscodes=[]
for codigo in codigos:
    nuevoscodes.append(str(int(codigo)//100))
unicos=np.unique(nuevoscodes).tolist()
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta
rng = np.random.default_rng(12345)
rares=[3101,3201,3119,2301,2403,3002,3012,3108,4003,2202,4201,3003,3113,4002,3013,4007,4203,3303,4202,3306]


# A partir de un path al archivo csv y un mes en particular a observar, genera cadena en estados resumidos en forma de lista.
# La lista consiste en tripletas donde: el primer elemento es el estado actual, el segundo el próximo estado y el tercero el tiempo usado en el estado actual.

def markover(path,mes):    
    # recorremos todos los equipos
    
        
    dato=pd.DataFrame(pd.read_csv(path))

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
            # En particular guardará el último. Considerar que quizás el tiempo en la entrada debería ser la suma y no solo el tiempo en el último(?)
        if state1!=state2:
            statesMon.append((state1,state2,state3))  
    return statesMon





# A partir de un array de una función sierra decreciente, la transforma en una creciente con el máximo valor 1

def transform_saw_function(arr):
    arr = np.asarray(arr)
    reset_points = (arr == 0)  # Máscara donde están los 0s

    # encontramos los segmentos de crecimiento
    segment_ids = np.cumsum(reset_points)  

    normalized_time = np.zeros_like(arr, dtype=float)
    for segment in np.unique(segment_ids):
        mask = segment_ids == segment  
        values = arr[mask]
        normalized_time[mask] = np.linspace(0, 1, len(values))  # incremento lineal a 1

    return normalized_time



# A partir de una trayectoria del formato entregado por markover() y una lista de estados de interés,
# genera una línea de tiempo de "probabilidades" asociadas a la visita próxima a un estado de la lista.
    
def sampler_regressor(trayectoria,reparacion=[30]):
    trayectoria=np.array(trayectoria)
    noise=np.where(~np.isin(trayectoria[:,0], rares))[0] # Filtro de estados raros
    trayectoria=np.array(trayectoria)[noise]
    trayectoria[:,0]=(np.array(trayectoria)[:,0]//100)
    trayectoria[:,1]=(np.array(trayectoria)[:,1]//100)
    
    noise2=np.where(~np.isin(trayectoria[:,0], [22,23]))[0] # Filtramos estados que no aportan al futuro modelo.
    trayectoria=np.array(trayectoria)[noise2].tolist()

    
    cumsum=np.cumsum(np.array(trayectoria)[:,2])  # Generamos un array del tiempo total de la trayectoria
    time=np.arange(0,cumsum[-1]+1,0.2) # Generamos una línea de tiempo "continua" de toda la trayectoria, donde cada instante t de tiempo es un quinto de hora.
    values=np.zeros(time.shape)
    
    
    # Con el siguiente loop generamos una serie de tiempo a partir de la trayectoria. En cada instante t de tiempo el equipo 
    # se encuentra en el estado que le corresponde según la trayectoria.
    prev_time=0
    
    for i,activity_time in enumerate(cumsum):
        start_idx = np.searchsorted(time, prev_time, side='left')
        end_idx = np.searchsorted(time, activity_time, side='right')
        
        values[start_idx:end_idx] = trayectoria[i][0] #Los ticks en ese intervalo de tiempo son registrados en el estado correspondiente
        
        prev_time = activity_time
    
   
    positions=np.where(np.isin(np.array(values), reparacion))[0] # Encontramos las posiciones en la serie de tiempo donde se visitan los estados de interés
    cumsum2=np.cumsum(np.array(time))

    # Con el siguiente loop generamos un array donde en cada instante nos dice el tiempo restante para la reparación más proóxima
    target=[]
    for i in range(len(time)):
        repairs=cumsum2[positions]
        dist=(repairs-cumsum2[i])
        try:
            value=min(dist[dist>=0])
        except:
            value=np.nan
        target.append(value)

    # Con la función transform_saw_function convertimos el array del tiempo restante a una "probabilidad" en el tiempo
    
    label=transform_saw_function(target)
    return values,label  



# Funciona casi idéntico a sampler regressor, pero en vez de observar estados de interés, lo hace sobre posiciones aleatorias.
# Esto permite generar un "grupo control", que permite demostrar que el modelo no está solo aprendiendo el el comportamiento de la funcion objetivo,
# si no también la relacion entre los estados de la trayectoria.

def sampler_regressor_rand(trayectoria,reparacion=[30]):
    trayectoria=np.array(trayectoria)
    noise=np.where(~np.isin(trayectoria[:,0], rares))[0] # Filtro de estados raros
    trayectoria=np.array(trayectoria)[noise]
    trayectoria[:,0]=(np.array(trayectoria)[:,0]//100)
    trayectoria[:,1]=(np.array(trayectoria)[:,1]//100)
    
    noise2=np.where(~np.isin(trayectoria[:,0], [22,23]))[0] # Filtramos estados que no aportan al futuro modelo.
    trayectoria=np.array(trayectoria)[noise2].tolist()
    cumsum=np.cumsum(np.array(trayectoria)[:,2])
    time=np.arange(0,cumsum[-1]+1,0.2)
    values=np.zeros(time.shape)
    
    

    prev_time=0
    
    for i,activity_time in enumerate(cumsum):
        start_idx = np.searchsorted(time, prev_time, side='left')
        end_idx = np.searchsorted(time, activity_time, side='right')
        
        values[start_idx:end_idx] = trayectoria[i][0]
        
        prev_time = activity_time

    positions=np.where(np.isin(np.array(trayectoria)[:,0], reparacion))[0]
    rand_positions=rng.uniform(low=10, high=len(time), size=len(positions)).astype(int) # Generamos tantos puntos aleatorios como estados de interés en la trayectoria original
    cumsum2=np.cumsum(np.array(time))
    
    target=[]
    for i in range(len(time)):
        repairs=cumsum2[rand_positions]
        dist=(repairs-cumsum2[i])
        try:
            value=min(dist[dist>=0])
        except:
            value=np.nan
        target.append(value)
    
    label=transform_saw_function(target)
    return values,label  


#sampler_regressor pero el label corresponde solo a las reparaciones outlier

def sampler_regressor2(trayectoria,reparacion=[30]):
    trayectoria=np.array(trayectoria)
    noise=np.where(~np.isin(trayectoria[:,0], rares))[0] # Filtro de estados raros
    trayectoria=np.array(trayectoria)[noise]
    trayectoria[:,0]=(np.array(trayectoria)[:,0]//100)
    trayectoria[:,1]=(np.array(trayectoria)[:,1]//100)
    
    noise2=np.where(~np.isin(trayectoria[:,0], [22,23]))[0] # Filtramos estados que no aportan al futuro modelo.
    trayectoria=np.array(trayectoria)[noise2].tolist()
    cumsum=np.cumsum(np.array(trayectoria)[:,2])
    time=np.arange(0,cumsum[-1]+1,0.2)
    values=np.zeros(time.shape)
    
    prev_time=0
    
    for i,activity_time in enumerate(cumsum):
        start_idx = np.searchsorted(time, prev_time, side='left')
        end_idx = np.searchsorted(time, activity_time, side='right')
        
        values[start_idx:end_idx] = trayectoria[i][0]
        
        prev_time = activity_time
        
    indices=np.where(np.isin(np.array(values), reparacion))[0]

    # Determinamos los intervalos donde ocurren los estados de interés y los marcamos con un 1
    y=np.zeros(len(values))
    y[indices]=y[indices]+1
    timeline = np.array([0]+y.tolist())

    start_indices = np.where(np.diff(timeline) == 1)[0]+1  # comienzo de los 1s
    end_indices = np.where(np.diff(timeline) == -1)[0]  # fin de los 1s

    representantes=[0]
    for (idx,idy) in zip(start_indices,end_indices): # En este loop vemos el punto medio de cada intervalo como su representante
        representantes.append(int((idx+idy)//2))
    representantes.append(len(timeline))
    
    numbers=np.diff(np.array(representantes))
    dist = np.array([min(numbers[i],numbers[i + 1]) for i in range(len(numbers) - 1)]) # esto nos dará la distancia mínima entre cada representante. Si es un outlier la distancia
                                                                                        # mínima debería ser grande.

    chosen=dist>=np.mean(dist)+np.std(dist) # Threshold un poco arbitrario, pero es estándar estadísticamente hablando
    
    # Marcamos los estados outlier en el loop para que luego puedan ser transformados en una funcion de probabilidad
    y2=np.zeros(len(values))
    for (chos,start,end) in zip(chosen,start_indices,end_indices):
        if chos==True:
  
            y2[start:end+1]=1

    positions=np.where(y2==1)[0]
    # Aquí sigue igual que antes...
    
    cumsum2=np.cumsum(np.array(time))
    
    target=[]
    for i in range(len(time)):
        repairs=cumsum2[positions]
        distan=(repairs-cumsum2[i])
        try:
            value=min(distan[distan>=0])
        except:
            value=np.nan
        target.append(value)
    
    label=transform_saw_function(target)

    return values,label




    

# A partir de los resultados de un sampler regressor, los procesa para que una LSTM pueda recibirlos
# Los valores de window_size y stride son relevantes a la hora de tunear los modelos.    

def sequence_generator(data,labels, window_size=240,stride=120):
    sequence_data, sequence_labels = [], []
    possible_states = ['10', '11', '20', '21', '24', '30', '31', '32', '33', '40'] #Cambiar a gusto
    
    # Generamos una versión One-Hot-Encoded de los estados
    demo_df = pd.DataFrame(data).astype(int)
    demo_df[0] = pd.Categorical(demo_df[0].astype(str), categories=possible_states)
    data_np = pd.get_dummies(demo_df[0]).reindex(columns=possible_states, fill_value=0).values

    
    labels_np = np.array(labels)
    # Creamos ventanas para el modelo de tamaño window_size. 240 ticks /5 son 48 horas de observación´.
    # El stride define el tiempo entre las ventanas. Si stride>window_size las ventanas no se solapan. 
    # Para un mejor entrenamiento es mejor que el solapamiento sea máximo, así se capturan pequeñas diferencias
    # durante el entrenamiento. Se invita a probar con distintos strides.
    
    if len(data_np) >= window_size:
        for j in range(0,len(data_np) - window_size + 1,stride):
            window = data_np[j:j + window_size]  
            label = labels_np[j + window_size ]  # el label es el valor de la función al final de la ventana
            sequence_data.append(np.transpose(window))
            sequence_labels.append(label)

    X, y = np.array(sequence_data), np.array(sequence_labels)
    valid_mask = ~np.isnan(y) # Eliminamos posibles nan values
    return X[valid_mask], y[valid_mask]


# Funcionamiento similar a sequence_generator, pero transforma los labels continuos en algo categórico. 
# Si la ventana contiene una reparación, vale 1. Si no, 0.

def sequence_generator2(data,labels, window_size=240,stride=10):
    sequence_data, sequence_labels = [], []
    possible_states = ['10', '11', '20', '21', '24', '30', '31', '32', '33', '40']

    demo_df = pd.DataFrame(data).astype(int)
    demo_df[0] = pd.Categorical(demo_df[0].astype(str), categories=possible_states)
    data_np = pd.get_dummies(demo_df[0],dtype=float).reindex(columns=possible_states, fill_value=0).values

    labels_np = np.array(labels)

    y=[]
    if len(data_np) >= window_size:
        for j in range(0,len(data_np) - window_size + 1,stride):
            window = data_np[j:j + window_size]  
            futuro = labels_np[j:j + window_size - 1]
            if 1 in futuro:
                y.append(1)
            else:
                y.append(0)
            sequence_data.append(np.transpose(window))
            
    X, y = np.array(sequence_data), np.array(y)
    valid_mask = ~np.isnan(y)
    return X[valid_mask], y[valid_mask]

def monotonicity_loss3():
    return 1


def moving_average(data, window_size=5):
    
    a=np.convolve(data, np.ones(window_size)/window_size, mode='same')
    return gaussian_filter1d(a, sigma=4)

