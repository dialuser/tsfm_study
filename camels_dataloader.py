#author: alex sun
#date: 10/31/2025
#purpose: clean up for 3H AR LSTM model
#
import os,sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import pickle as pkl
from tstutils.timefeatures import time_features

from torch.utils.data import Dataset, DataLoader, TensorDataset,ConcatDataset
from camels_dataloader_daily import load_forcing

def __read_gauge_info(path):
    """ Read gauge static data
        Modified from https://github.com/kratzert/lstm_for_pub/blob/master/extract_benchmarks.py
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

def getUSGSData(cfg, gageid, area, mode='test'):
    """Load 3-hr data and truncate to the start/end dates
    gageid, a list
    """
    if mode=='train':
        startDate = cfg.model.train_start_date
        endDate   = cfg.model.train_end_date
    else:
        startDate = cfg.model.start_date
        endDate   = cfg.model.end_date

    gagedf = None
    if mode == 'train':
        filename = f'{cfg.rootdir}/{cfg.camel_data_repo}/{int(gageid)}/usgs_data_iv_train.pkl'
    else:
        filename = f'{cfg.rootdir}/{cfg.camel_data_repo}/{int(gageid)}/usgs_data_iv.pkl'

    if os.path.exists(filename):
        usgsDataDict = pd.read_pickle(filename)
        alldates = pd.date_range(start=startDate, end=endDate, freq='3h').tz_localize('UTC')
        
        #aggregate data to 3H
        for key, df in usgsDataDict.items():
            if int(key) == gageid:
                df.index = pd.to_datetime(df.index,utc=True)    
                #asun03102024, replace 3H with 3h
                df = df.resample('3h').mean()

                #asun 0607, truncate the df if necessary
                mask =  (df.index >= pd.to_datetime(startDate).tz_localize('UTC')) & (df.index <= pd.to_datetime(endDate).tz_localize('UTC'))
                gagedf = df.loc[mask].copy()
                if gagedf is None or gagedf.empty:
                    continue                
                if not len(gagedf)==len(alldates):
                    gagedf = gagedf.reindex(alldates)
                # normalize discharge from cubic feet per second to mm per 3h
                gagedf.Q = 28316846.592 * gagedf.Q * 3600*3 / (area * 10**6)  
                
                gagedf['timestamp'] = pd.to_datetime(gagedf.index)
                gagedf['item_id']   = gageid
                break
      
    return gagedf

def loadCAMELS_List(cfg):
    """Load a list of 531 stations"""
    #basinlistfile = f'{cfg.nwm_root_dir}/camels/basin_dataset_public_v1p2/basin_metadata/gauge_information.csv'
    basinlistfile = f'{cfg.nwm_root_dir}/camels/basinlist531.txt'
    df_basinset = pd.read_csv(basinlistfile, header=None)
    df_basinset.columns=['gage_id']

    return df_basinset

def formDataSet(gageid, 
                df_raw:pd.DataFrame, 
                lookback:int, 
                forecast_len:int, 
                label_len:int = 0,
                window_stride:int=1,
                returnGageID:bool=False, 
                timeenc:int=1, 
                freq:str='3h', 
                include_forcing:bool=False,
                allow_missing:bool=False):
    """Form input/target data pairs for uni-variate autoregression
    window stride, the stride between each moving window
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
    
    data_x = df_raw[['Q']]
    data_y = df_raw[['Q']]
    X = []
    Y = []
    X_mark = []
    Y_mark = []

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
        print (f"usable data for {gageid}: {X.shape[0]}")
        if returnGageID:
            GAGEID = torch.zeros((X.shape[0],1), dtype=torch.long)+gageid
            dataset = TensorDataset(X, Y, X_mark, Y_mark, GAGEID)
        else:
            dataset = TensorDataset(X, Y, X_mark, Y_mark)
    else:
        print ('no data')
        dataset = None
    return dataset


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
       
        input_df=[]   
        for gage_id in tqdm(watershedList):   
            _, basin_area = load_forcing(self.cfg, forcingType='nldas', basin=gage_id, mode=mode)
            gageDF = getUSGSData(self.cfg, gage_id, area=basin_area, mode=mode)
            if gageDF is None or gageDF.empty:
                print (f'skipping {gage_id}')
                continue  
                    
            print (f'Processing {gage_id}')
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
            input_df.append(gageDF)

        if self.cfg.model.normalization:
            big_df = pd.concat(input_df)
            stats_file = 'camels_3Hstats'
            if self.cfg.model.log_transform:
                stats_file += f'_{self.cfg.model.transform_method}'
            stats_file +='.pkl'
            if mode=='train':
                if regen_stats:            
                    qarr = big_df['Q'].values      
                    q_mean = np.nanmean(qarr)
                    q_std = np.nanstd(qarr)
                    pkl.dump({'qmean':q_mean, 'qstd':q_std}, open(stats_file, 'wb'))
                    del big_df

            dct = pkl.load(open(stats_file, 'rb'))
            q_mean = dct['qmean']
            q_std  = dct['qstd']
            print ('Q mean', q_mean, q_std)            

            for df in input_df:
                qarr = df['Q'].values 
                qarr = (qarr - q_mean) / q_std
                df.Q = qarr        
        
        if mode in ['train', 'val']:
            returnGageID  = False
        else:
            returnGageID  = True

        datasets = []
        for df in tqdm(input_df): 
            gage_id = df['item_id'].values[0]
            dataset = formDataSet(gageid=gage_id, 
                                  df_raw=df, 
                                  lookback=self.cfg.model.seq_len, 
                                  forecast_len=self.cfg.model.pred_len, 
                                  window_stride=1,
                                  returnGageID=returnGageID)
            datasets.append( dataset)        
           
        return datasets

    def getDataLoader(self, mode, reload=False, regen_stats=False):
        if reload:
            datasets = self.__genDataSets(mode, regen_stats)
            torch.save(datasets, f'{mode}_dataset_{self.cfg.model.seq_len}_{self.cfg.model.pred_len}.pt')
        else:
            datasets = torch.load(f'{mode}_dataset_{self.cfg.model.seq_len}_{self.cfg.model.pred_len}.pt')
        
        total_sample_size = 0
        valid_datasets = []
        for ds in datasets:
            if not ds is None:
                total_sample_size+= len(ds)
                valid_datasets.append(ds)
        print ('total sample size', total_sample_size)
        if mode == 'train':
            batch_size = self.cfg.model.batch_size
            combined_dataset = ConcatDataset(valid_datasets)
            
            return DataLoader(combined_dataset, 
                              batch_size=batch_size,
                              num_workers=self.cfg.model.num_workers,
                              shuffle=True,
                              drop_last = True)
        elif mode == 'val':
            batch_size = self.cfg.model.batch_size

            # Create DataLoader
            combined_dataset = ConcatDataset(valid_datasets)
            return DataLoader(combined_dataset, 
                              batch_size=batch_size,
                              num_workers=self.cfg.model.num_workers,
                              shuffle=False,
                              drop_last = True
                              )

        elif mode == 'test':
            batch_size = self.cfg.model.batch_size

            combined_dataset = ConcatDataset(valid_datasets)            
            return DataLoader(
                            combined_dataset,
                            batch_size= batch_size,
                            num_workers=self.cfg.model.num_workers,
                            shuffle= False,
                            drop_last = False
                            )            
        else: 
            raise ValueError("Invalid mode")


