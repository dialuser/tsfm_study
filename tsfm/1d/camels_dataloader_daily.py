#author: alex sun
#date: 09012025
#purpose: daily streamflow
#date 10072025, clean up
#date 10292025, revised Dataloader 
#===================================================================================
import os,sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import pickle as pkl
from pathlib import Path, PosixPath
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader, TensorDataset,Sampler,ConcatDataset
from tstutils.timefeatures import time_features

def __read_gauge_info(path):
    """ Read gauge static data
        Modified from https://github.com/kratzert/lstm_for_pub/blob/master/extract_benchmarks.py
        Params:
        -------
        path: path of the gauge_info csv file
        Returns:
        -------
        gauge_info: dictionary of camels basin attributes
    """

    gauge_info = pd.read_csv(path)
    gauge_info.columns=['huc2','gauge_id','gauge_name','lat','lng','drainage_area']

    gauge_info['gauge_str'] = gauge_info['gauge_id']
    gauge_info['gauge_str'] = gauge_info['gauge_str'].apply(lambda x: '{0:0>8}'.format(x))

    gauge_info['gauge_id'] = gauge_info['gauge_id'].apply(pd.to_numeric)
    gauge_info['lat'] = gauge_info['lat'].apply(pd.to_numeric)
    gauge_info['lng'] = gauge_info['lng'].apply(pd.to_numeric)
    return gauge_info

def getStaticAttr(cfg):
    """ Load static attributes of all 531 basins
        Assume the camels data are in camels 
    Params
    ------
    cfg: configuration yaml 
    """
    basinlistfile = f'{cfg.nwm_root_dir}/camels/basinlist531.txt'
    df_basinset = pd.read_csv(basinlistfile, header=None)
    df_basinset.columns=['gauge_id']

    # --- Metadata and Catchment Characteristics ---------------------------

    # The purpose of loading this metadata file is to get huc and basin IDs for
    # constructing model output file names.
    # we also need the gauge areas for normalizing NWM output.

    # load metadata file (with hucs)
    meta_df = __read_gauge_info(f'{cfg.nwm_root_dir}/camels/basin_dataset_public_v1p2/basin_metadata/gauge_information.csv')
    assert meta_df['gauge_id'].is_unique  # make sure no basins or IDs are repeated
    # concatenate catchment characteristics with meta data
    meta_df = meta_df.round({
        'lat': 5,
        'lng': 5
    })  # latitudes and longitudes should be to 5 significant digit

    #get subbasins

    meta_df = df_basinset.join(
            meta_df.set_index('gauge_id'),
            on='gauge_id')  
    # load characteristics file (with areas)
    rootloc = f'{cfg.nwm_root_dir}/camels/camels_attributes_v2.0/'  # catchment characteristics file name
    fnames = ['camels_clim.txt','camels_geol.txt','camels_hydro.txt','camels_soil.txt','camels_topo.txt','camels_vege.txt']
    static_df = None
    for afile in fnames:
        fname = '/'.join([rootloc, afile])
        print ('processing', fname)
        char_df = pd.read_table(fname, delimiter=';', dtype={'gauge_id': int})  # read characteristics file

        assert char_df['gauge_id'].is_unique  # make sure no basins or IDs are repeated

        char_df = char_df.round({'gauge_lat': 5, 'gauge_lon': 5})
        #assert meta_df['gauge_id'].equals(
        #    char_df['gauge_id'])  # check catchmenet chars & metdata have the same basins
        #assert meta_df['lat'].equals(char_df['gauge_lat'])  # check that latitudes and longitudes match
        #assert meta_df['lng'].equals(char_df['gauge_lon'])
        if static_df is None:
            static_df = char_df.join(
            meta_df.set_index('gauge_id'),
            on='gauge_id',how='right')  # turn into a single dataframe (only need huc from meta)
        else:
            static_df = char_df.join(
            static_df.set_index('gauge_id'),
            on='gauge_id', how='right')  # turn into a single dataframe (only need huc from meta)

    nBasins = static_df.shape[0]  # count number of basins

    print ('number of basins', nBasins)

    return static_df

def getSubSet(allDF):
    """Return a subset of static attribute dataframe
    Reference: Nearing 2019 WRR paper, Table 1
    
    Params:
    ------
    allDF, dataframe containing all static attr 
    """
    colnames = [
        'p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow',
        'high_prec_freq', 'high_prec_dur','low_prec_freq', 'low_prec_dur', 'elev_mean',
        'slope_mean', 'area_gages2', 'frac_forest', 'lai_max', 'lai_diff',
        'gvf_max', 'gvf_diff','soil_depth_pelletier', 'soil_depth_statsgo','soil_porosity',
        'soil_conductivity','max_water_content', 'sand_frac', 'silt_frac','clay_frac',
        'geol_permeability', 'carbonate_rocks_frac',
    ]
    return  allDF[colnames]


def getUSGSData(cfg, gageid: str, mode:str, area:float) -> pd.Series:
    """[summary]
    Parameters
    ----------
    cfg: configuration yaml file    
    gageid : str
        8-digit USGS gauge id

    Returns
    -------
    pd.Series
        A Series containing the discharge values.
    Raises
    ------
    RuntimeError
        If no discharge file was found.
    """
    if mode=='train':
        startDate = cfg.model.train_start_date
        endDate   = cfg.model.train_end_date
    elif mode == 'val':
        startDate = cfg.model.val_start_date
        endDate   = cfg.model.val_end_date
    else:
        startDate = cfg.model.start_date
        endDate   = cfg.model.end_date

    camels_root = PosixPath(cfg.nwm_root_dir)
    camels_root = camels_root / 'camels/basin_dataset_public_v1p2'
    discharge_path = camels_root / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    #convert gageid to 8-digit string id
    gageid = f'{gageid:08d}'
    file_path = [f for f in files if f.name[:8] == gageid]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {gageid} at {file_path}')
    else:
        file_path = file_path[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'Q', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    #count the number of data records
    full_daterng  = pd.date_range(start='1980-01-01', end='2014-12-31', freq='1d', inclusive='both')
    if len(df)< len(full_daterng):
        df = df.reindex(full_daterng)
        df["Year"] = df.index.year
        df['Mnth'] = df.index.month
        df['Day']  = df.index.day

    # normalize discharge from cubic feed per second to mm per day
    df.Q = 28316846.592 * df.Q * 86400 / (area * 10**6)        
    
    #asun 0607, truncate the df if necessary
    if mode in ['train', 'val', 'test']:
        mask =  (df.index >= pd.to_datetime(startDate)) & (df.index <= pd.to_datetime(endDate))
        gagedf = df.loc[mask]
        return gagedf
    else:    
        return df

def load_forcing(cfg, forcingType: str, basin: int, mode:str='train') -> Tuple[pd.DataFrame, int]:
    """Load Maurer forcing data from text files.
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id
    forcingType: str
        type of forcing data 
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the Maurer forcing
    area: int
        Catchment area (read-out from the header of the forcing file)
    Raises
    ------
    RuntimeError
        If not forcing file was found.
    """
    #    forcing_path = camels_root / 'basin_mean_forcing' / 'maurer_extended'
    #    forcing_path = camels_root / 'basin_mean_forcing' / 'nldas'
    #forcing_path = camels_root / 'basin_mean_forcing' / 'nldas_extended'
    camels_root = PosixPath(cfg.nwm_root_dir)
    camels_root = camels_root / 'camels/basin_dataset_public_v1p2'
    if forcingType=='maurer':
        forcing_path = camels_root / 'basin_mean_forcing' / 'maurer_extended'
    elif forcingType=="nldas":
        forcing_path = camels_root / 'basin_mean_forcing' / 'nldas_extended'
    else:
        raise RuntimeError("not a valid forcing data type")
    
    #convert to 8-digit string id
    basin = f'{basin:08d}'
    
    files = list(forcing_path.glob('**/*_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    df = pd.read_csv(file_path, sep='\s+', header=None, skiprows=4)
    #standardize column names
    #note some of the original files have missing column headers
    #e.g., basin_dataset_public_v1p2\basin_mean_forcing\maurer\03\02108000_lump_maurer_forcing_leap.txt02108000
    #
    df.columns = ['Year', 'Mnth', 'Day', 'Hr', 'Dayl(s)', 'PRCP', 'SRAD',
       'SWE', 'Tmax', 'Tmin', 'Vp']
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    
    # load area from header
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    if mode=='train':
        startDate = cfg.model.train_start_date
        endDate   = cfg.model.train_end_date    
    elif mode=='val':
        startDate = cfg.model.val_start_date
        endDate   = cfg.model.val_end_date
    else:
        startDate = cfg.model.start_date
        endDate   = cfg.model.end_date

    if mode in ['train', 'val', 'test']:
        mask =  (df.index >= pd.to_datetime(startDate)) & (df.index <= pd.to_datetime(endDate))
        df = df.loc[mask]
        return df, area
    else:    
        return df, area


def getLogP3Peaks(cfg, gage_id, T, min_years=10, log_base=10):
    from scipy.stats import pearson3, skew
    """
    Compute T-year return level(s) using Log-Pearson Type III.
    
    Parameters
    ----------
    peaks : array-like
        Annual maximum series (one value per year).
    T : scalar or array-like
        Return period(s) in years. e.g., 2, 10, 100 or [2,10,100]
    log_base : {10, np.e} (default 10)
        Log base to use. Default is 10 (common in hydrology).
    
    Returns
    -------
    Q_T : ndarray
        Return level(s) corresponding to T (same shape as T).
    info : dict
        Dictionary with keys 'mean_log', 'std_log', 'skew_log', 'n' for diagnostics.
    """
    mode = 'train'
    _, basin_area = load_forcing(cfg, forcingType='nldas', basin=gage_id, mode=mode)
    gageDF = getUSGSData(cfg, gage_id, mode=mode, area=basin_area)
    peaks = gageDF["Q"].resample("A-SEP").max().values

    if peaks.size == 0:
        raise ValueError("peaks array is empty.")

    # Quality checks
    if peaks.size < min_years:
        raise ValueError(f"Insufficient data: {peaks.size} years < {min_years} minimum")

    if np.any(peaks <= 0):
        raise ValueError("Log-Pearson III requires positive flows. Found zero or negative values.")

    # Convert to log space
    if log_base == 10:
        y = np.log10(peaks)
        inv = lambda z: 10**z
    elif log_base == np.e:
        y = np.log(peaks)
        inv = np.exp
    else:
        # generic base b: y = log_b(x) = ln(x)/ln(b)
        ln_b = np.log(log_base)
        y = np.log(peaks) / ln_b
        inv = lambda z: (log_base**z)

    # Use loc=mean_y, scale=std_y, skew=skew_y
    # non-exceedance probability for return period T:
    n = y.size
    mean_y = np.mean(y)
    std_y = np.std(y, ddof=1)  # sample standard deviation
    # sample (unbiased) skewness: scipy.stats.skew with bias=False
    skew_y = skew(y, bias=False)
    
    # Calculate flood magnitudes
    T_arr = np.atleast_1d(T)
    if np.any(T_arr <= 1):
        raise ValueError("Return period T must be > 1 year.")
    p = 1.0 - 1.0 / T_arr  # non-exceedance probability

    # Fit a Pearson III in log-space using the sample moments:    
    # For pearson3.ppf, shape parameter is 'skew'
    y_T = pearson3.ppf(p, skew_y, loc=mean_y, scale=std_y)
    Q_T = inv(y_T)
    
    info = {'n': int(n), 'mean_log': mean_y, 'std_log': std_y, 'skew_log': skew_y}
    return Q_T if Q_T.size>1 else Q_T[0], info



def loadCAMELS_List(cfg):
    """Load a list of 531 stations
    cfg: yaml configuration file
    """
    #basinlistfile = f'{cfg.nwm_root_dir}/camels/basin_dataset_public_v1p2/basin_metadata/gauge_information.csv'
    basinlistfile = f'{cfg.nwm_root_dir}/camels/basinlist531.txt'
    df_basinset = pd.read_csv(basinlistfile, header=None)
    df_basinset.columns=['gage_id']

    return df_basinset

def formDataSet(gageid:int,
                df_raw:pd.DataFrame, 
                lookback:int, 
                forecast_len:int, 
                label_len:int,
                window_stride:int=1, 
                timeenc:int=1, 
                freq:str='D', 
                returnGageID:bool=False, 
                include_forcing:bool=True,
                predict_mode:str = 'forecast',
                allow_missing = False):
    """Form input/target torch dataset pairs 
    Params
    ------
    gageid, gauge id (integer not 8-digit str)
    df_raw, dataframe containing forcing&Q
    lookback, length of the lookback window
    forecast_len: length of the forecast window
    window_stride: stride between consecutive windows
    timeend: time encoding method to use
    freq: freq of data
    returnGageID: True to return gageid as part of the __getitem__
    include_forcing: True to append forcing to the dataset
    predict_mode, 'regress', 'forecast'
    allow_missing, allow missing data in forcing only
    """
    df_stamp = df_raw[['timestamp']]    
    if timeenc == 0:
        df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['timestamp'], axis=1).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    else:
        raise ValueError("invalid timestamp encoding method")
    
    X = []
    Y = []
    X_mark = []
    Y_mark = []
    
    if predict_mode == 'regress':
        #shift Q by 1 day
        df_raw['Q'] = df_raw['Q'].shift(periods=1)
    
    if include_forcing:
        data_x = df_raw[['PRCP', 'SRAD', 'Tmax', 'Tmin', 'Vp']]
    else:
        data_x = df_raw[['Q']]

    data_y = df_raw[['Q']]
    for i in range(lookback, df_raw.shape[0]-forecast_len+1, window_stride):
        s_begin, s_end = (i-lookback, i)
        r_begin = s_end-label_len
        r_end   = r_begin+label_len+forecast_len

        seq_x = data_x.iloc[s_begin:s_end].values
        seq_y = data_y.iloc[r_begin:r_end].values
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        if not allow_missing:
            if (not np.isnan(seq_x).sum()>0) and (not np.isnan(seq_y).sum()>0):
                X.append(seq_x)
                Y.append(seq_y)
                X_mark.append(seq_x_mark)
                Y_mark.append(seq_y_mark)
        else:
            if not np.isnan(seq_y).sum()>0:
                X.append(seq_x)
                Y.append(seq_y)
                X_mark.append(seq_x_mark)
                Y_mark.append(seq_y_mark)

    if X:
        X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)
        Y = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)
        X_mark = torch.tensor(np.stack(X_mark, axis=0), dtype=torch.float32)
        Y_mark = torch.tensor(np.stack(Y_mark, axis=0), dtype=torch.float32)
        print (f"usable data for {gageid}: {X.shape[0]}, {Y.shape[0]}")
        if returnGageID:
            GAGEID = torch.zeros((X.shape[0],1), dtype=torch.long)+gageid
            dataset = TensorDataset(X, Y, X_mark, Y_mark, GAGEID)
        else:
            dataset = TensorDataset(X, Y, X_mark, Y_mark)
    else:
        print ('no data')
        dataset = None
    return dataset

# --- create Fourier terms to capture weekly/annual seasonality ---
def fourier_series(index, period, K):
    t = np.arange(len(index))
    mat = {}
    for k in range(1, K+1):
        mat[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
        mat[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(mat, index=index)

def gen_exog(gageDF):
    daily = 1
    weekly = 7
    annual = 365  # only include if you have many years
    exog_weekly = fourier_series(gageDF.index, period=weekly, K=2)   # captures weekly shape
    exog_annual = fourier_series(gageDF.index, period=annual, K=3)
    exog_data = pd.concat([exog_weekly], axis=1)
    gageDF = pd.concat([gageDF, exog_data], axis=1)

    return gageDF


class AllDataloader():
    """Load data from all camels gages """
    def __init__(self, cfg):
        self.cfg = cfg

        #use all watersheds
        self.watershedlist_train =  loadCAMELS_List(cfg)['gage_id'].to_list()       
        self.watershedlist_val   =  self.watershedlist_train
        self.watershedlist_test  =  self.watershedlist_train
                    
    def __genDataSets(self, mode='train', regen_stats=False):
        """Generate dataset for mode, mode='train', 'val' or 'test'
        """                    
        if mode == 'train':
            watershedList = self.watershedlist_train
        elif mode == 'val':
            watershedList = self.watershedlist_val
        elif mode == 'test':
            watershedList = self.watershedlist_test

        print (f'len of watershedlist for {mode}', len(self.watershedlist_train))
        
        datasets=[]   
        
        input_df = []       
         
        for gage_id in tqdm(watershedList):     
            forcingDF, basin_area = load_forcing(self.cfg, forcingType='nldas', basin=gage_id, mode=mode)
            forcingDF = forcingDF[['PRCP', 'SRAD', 'Tmax', 'Tmin', 'Vp']]

            gageDF  = getUSGSData(self.cfg, gage_id, mode=mode, area=basin_area)

            if gageDF is None or gageDF.empty:
                print (f'skipping {gage_id}')
                continue  
                    
            print (f'Processing {gage_id}', gageDF.shape)
            arr=gageDF['Q'].to_numpy()

            #mask invalid data
            arr[arr<0] = np.nan
            if self.cfg.model.log_transform:
                if self.cfg.model.transform_method=='log':
                    gageDF.Q = np.log1p(arr)
                elif self.cfg.model.transform_method=='feng':                    
                    gageDF.Q = np.log10(np.sqrt(arr) + 0.1)
                else:
                    raise ValueError("Invalid transform method")                    
            else:
                gageDF.Q = arr                    

            gageDF = gageDF.drop(['flag', 'basin', 'Day'], axis=1)

            if self.cfg.model.add_forcing:
                gageDF = pd.concat([gageDF, forcingDF], axis=1)
            
            gageDF['timestamp'] = pd.to_datetime(gageDF.index)    
            gageDF['item_id']   = gage_id

            input_df.append(gageDF)

        if self.cfg.model.normalization:
            big_df = pd.concat(input_df)
            stats_file = 'camels_dailystats'
            if self.cfg.model.log_transform:
                stats_file += f'_{self.cfg.model.transform_method}'
            stats_file +='.pkl'

            if mode=='train':
                if regen_stats:            
                    #get forcing stats
                    if self.cfg.model.add_forcing:
                        forcingarr = big_df[['PRCP', 'SRAD', 'Tmax', 'Tmin', 'Vp']].values
                        forcing_means = np.nanmean(forcingarr, axis=0)
                        forcing_stds  = np.nanstd(forcingarr, axis=0)
                    qarr = big_df['Q'].values      
                    q_mean = np.nanmean(qarr)
                    q_std = np.nanstd(qarr)
                    if self.cfg.model.add_forcing:
                        pkl.dump({'mean':forcing_means, 'std': forcing_stds, 'qmean':q_mean, 'qstd':q_std}, open(stats_file, 'wb'))
                    else:
                        pkl.dump({'qmean':q_mean, 'qstd':q_std}, open(stats_file, 'wb'))
                    del big_df, 

            dct = pkl.load(open(stats_file, 'rb'))
            if self.cfg.model.add_forcing:
                forcing_means = dct['mean']
                forcing_stds  = dct['std']
                print ('Forcing mean', forcing_means, forcing_stds)
            q_mean = dct['qmean']
            q_std  = dct['qstd']
            print ('Q mean', q_mean, q_std)            
            

            for df in input_df:
                qarr = df['Q'].values 
                qarr = (qarr - q_mean) / q_std
                df.Q = qarr        

                if self.cfg.model.add_forcing:
                    forcingarr = df[['PRCP', 'SRAD', 'Tmax', 'Tmin', 'Vp']].values
                    forcingarr  = (forcingarr - forcing_means) / forcing_stds
                    df[['PRCP', 'SRAD', 'Tmax', 'Tmin', 'Vp']] = forcingarr

        if mode in ['train', 'val']:
            returnGageID  = False
        else:
            returnGageID  = True

        for df in input_df:            
            dataset = formDataSet(gageid=df['item_id'].values[0], 
                                df_raw = df, 
                                lookback=self.cfg.model.seq_len, 
                                forecast_len=self.cfg.model.pred_len,
                                label_len=0, 
                                window_stride=1, 
                                returnGageID=returnGageID,
                                predict_mode=self.cfg.model.predict_mode,
                                include_forcing=self.cfg.model.add_forcing)
            if not dataset is None:
                datasets.append(dataset)
           
        return datasets

    def getDataLoader(self, mode, reload=False, regen_stats=False):
        if reload:
            print (f'loading {mode} datasets')
            datasets = self.__genDataSets(mode, regen_stats)
            print (f'saving {mode} datasets')
            torch.save(datasets, f'{mode}_dailydataset_{self.cfg.model.seq_len}_{self.cfg.model.pred_len}.pt')
        else:
            datasets = torch.load(f'{mode}_dailydataset_{self.cfg.model.seq_len}_{self.cfg.model.pred_len}.pt', weights_only=False)

        lens = [len(ds) for ds in datasets]
        print ('total sample size', np.sum(lens))
        if mode == 'train':
            batch_size = self.cfg.model.batch_size
            # Create DataLoader
            combined_dataset = ConcatDataset(datasets)
            
            return DataLoader(combined_dataset, 
                              batch_size=batch_size,
                              num_workers=self.cfg.model.num_workers,
                              shuffle=True,
                              drop_last = True)
        elif mode == 'val':
            batch_size = self.cfg.model.batch_size

            # Create DataLoader
            combined_dataset = ConcatDataset(datasets)
            return DataLoader(combined_dataset, 
                              batch_size=batch_size,
                              num_workers=self.cfg.model.num_workers,
                              shuffle=False,
                              drop_last = True
                              )

        elif mode == 'test':
            batch_size = self.cfg.model.batch_size_test

            combined_dataset = ConcatDataset(datasets)            
            return DataLoader(
                            combined_dataset,
                            batch_size= batch_size,
                            num_workers=self.cfg.model.num_workers_test,
                            shuffle= False,
                            drop_last = False
                            )        
        else: 
            raise ValueError("Invalid mode")


