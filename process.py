import numpy as np
import pandas as pd

rng = np.random.default_rng(12345)

def transform_saw_function(arr):
    arr = np.asarray(arr)  # Ensure input is a NumPy array
    reset_points = (arr == 0)  # Boolean mask for reset points

    # Create an increasing index for each segment
    segment_ids = np.cumsum(reset_points)  # Assigns a unique ID to each segment

    # Get normalized time within each segment
    normalized_time = np.zeros_like(arr, dtype=float)
    for segment in np.unique(segment_ids):
        mask = segment_ids == segment  # Select only this segment
        values = arr[mask]
        normalized_time[mask] = np.linspace(0, 1, len(values))  # Linearly increasing

    return normalized_time


def sampler_regressor(trayectoria,reparacion=[30]):
    noise=np.where(~np.isin(np.array(trayectoria)[:,0], [22,23]))[0]
    trayectoria=np.array(trayectoria)[noise].tolist()
    positions=np.where(np.isin(np.array(trayectoria)[:,0], reparacion))[0]
    cumsum=np.cumsum(np.array(trayectoria)[:,2])
    time=np.arange(0,cumsum[-1]+1,0.2)
    values=np.zeros(time.shape)
    label=np.zeros(time.shape)
    
    

    prev_time=0
    
    for i,activity_time in enumerate(cumsum):
        start_idx = np.searchsorted(time, prev_time, side='left')
        end_idx = np.searchsorted(time, activity_time, side='right')
        
        values[start_idx:end_idx] = trayectoria[i][0]
        
        prev_time = activity_time
    
    rand_positions=rng.uniform(low=10, high=len(time), size=len(positions)).astype(int)
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




def sampler_regressor2(trayectoria,reparacion=[30]):
    noise=np.where(~np.isin(np.array(trayectoria)[:,0], [22,23]))[0]
    trayectoria=np.array(trayectoria)[noise].tolist()
    positions=np.where(np.isin(np.array(trayectoria)[:,0], reparacion))[0]
    cumsum=np.cumsum(np.array(trayectoria)[:,2])
    time=np.arange(0,cumsum[-1]+1,0.2)
    values=np.zeros(time.shape)
    label=np.zeros(time.shape)
    
    

    prev_time=0
    
    for i,activity_time in enumerate(cumsum):
        start_idx = np.searchsorted(time, prev_time, side='left')
        end_idx = np.searchsorted(time, activity_time, side='right')
        
        values[start_idx:end_idx] = trayectoria[i][0]
        
        prev_time = activity_time
    
    positions=np.where(np.isin(np.array(values), reparacion))[0]
    cumsum2=np.cumsum(np.array(time))
    
    target=[]
    for i in range(len(time)):
        repairs=cumsum2[positions]
        dist=(repairs-cumsum2[i])
        try:
            value=min(dist[dist>=0])
        except:
            value=np.nan
        target.append(value)
    
    label=transform_saw_function(target)
    return values,label  

def sampler_regressor3(trayectoria,reparacion=[30]):
    noise=np.where(~np.isin(np.array(trayectoria)[:,0], [22,23]))[0]
    trayectoria=np.array(trayectoria)[noise].tolist()
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
    y=np.zeros(len(values))
    y[indices]=y[indices]+1
    timeline = np.array([0]+y.tolist())

    #######
    #print(timeline[0:30])
    start_indices = np.where(np.diff(timeline) == 1)[0]+1  # Start of 1s
    end_indices = np.where(np.diff(timeline) == -1)[0]  # End of 1s

    representantes=[0]
    for (idx,idy) in zip(start_indices,end_indices):
        representantes.append(int((idx+idy)//2))
    representantes.append(len(timeline))
    numbers=np.diff(np.array(representantes))
    dist = np.array([min(numbers[i],numbers[i + 1]) for i in range(len(numbers) - 1)])
    #print(len(dist))
    chosen=dist>=np.mean(dist)+np.std(dist)
    ####
    #print(chosen)
    #print(len(start_indices),start_indices)
    #print(len(end_indices),end_indices)
    y2=np.zeros(len(values))
    for (chos,start,end) in zip(chosen,start_indices,end_indices):
        if chos==True:
            #print("worked")
            #print(start,end)
            y2[start:end+1]=1
    #print(y2)
    positions=np.where(y2==1)[0]
    #print(positions)        
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
    

    

def sequence_generator(data,labels, window_size=240, reparacion=[30],stride=120):
    sequence_data, sequence_labels = [], []
    possible_states = ['10', '11', '20', '21', '24', '30', '31', '32', '33', '40']
    

    # Convert to DataFrame and ensure categorical states
    demo_df = pd.DataFrame(data).astype(int)
    demo_df[0] = pd.Categorical(demo_df[0].astype(str), categories=possible_states)
    data_np = pd.get_dummies(demo_df[0]).reindex(columns=possible_states, fill_value=0).values

    labels_np = np.array(labels)
    
    # Ensure sufficient data
    if len(data_np) >= window_size:
        for j in range(0,len(data_np) - window_size + 1,stride):
            window = data_np[j:j + window_size]  
            label = labels_np[j + window_size -1]  
            sequence_data.append(window)
            sequence_labels.append(label)
    
    # Convert to numpy arrays and filter NaN labels
    X, y = np.array(sequence_data), np.array(sequence_labels)
    valid_mask = ~np.isnan(y)
    return X[valid_mask], y[valid_mask]

def sequence_generator2(data,labels, window_size=240, reparacion=[30],stride=10):
    sequence_data, sequence_labels = [], []
    possible_states = ['10', '11', '20', '21', '24', '30', '31', '32', '33', '40']
    

    # Convert to DataFrame and ensure categorical states
    demo_df = pd.DataFrame(data).astype(int)
    demo_df[0] = pd.Categorical(demo_df[0].astype(str), categories=possible_states)
    data_np = pd.get_dummies(demo_df[0],dtype=float).reindex(columns=possible_states, fill_value=0).values

    labels_np = np.array(labels)
    # Ensure sufficient data
    y=[]
    if len(data_np) >= window_size:
        for j in range(0,len(data_np) - window_size + 1,stride):
            window = data_np[j:j + window_size]  
            futuro = labels_np[j:j + window_size - 1]
            if 1 in futuro:
                y.append(1)
            else:
                y.append(0)
            sequence_damodelta.append(window)
            
    
    # Convert to numpy arrays and filter NaN labels
    X, y = np.array(sequence_data), np.array(y)
    valid_mask = ~np.isnan(y)
    return X[valid_mask], y[valid_mask]

