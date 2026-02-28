#author: alex sun
#date: 06/03/2025
#rev date: 08/16/2025
#rev date: 08/28/2025, changed to new camels_dataloader.py
#Test zero-shot on camels
#conda env: tslm for sundial
#conda env: chronos for chronos
#conda env: moirai for moirai
#conda env: ibm for ttm
#command to run
#>python sundial_camels.py --config config_sundial_camels.yaml
#=======================================================================
import os,sys
import warnings

from typing import List,Dict
from pathlib import Path
import pickle as pkl
import pandas as pd
import numpy as np
import torch
from omegaconf import OmegaConf

from tqdm import tqdm
from transformers import AutoModelForCausalLM
from torch.utils.data import TensorDataset,DataLoader

import matplotlib.pyplot as plt
import hydrostats as Hydrostats
import hydrostats.ens_metrics as EM

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.common import ListDataset
except ImportError:
    print ('ListDataset not imported')

try:
    from chronos import BaseChronosPipeline
except ImportError:
    print ('BaseChronosPipeline not imported')

try: 
    from tsfm_public import TimeSeriesForecastingPipeline,TinyTimeMixerForPrediction, TimeSeriesPreprocessor
except ImportError:
    print ('TSFM public not imported')

from camels_dataloader import formDataSet,loadCAMELS_List,getUSGSData
from camels_utils import fdc_metrics
from camels_dataloader_daily import load_forcing

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print ('='*10, 'Device ', device)

def getMetrics(Pred, Tru, forecast_len, allRes=None):
    #calculate metrics
    KGE=[]
    NSE=[]
    CRPS=[]
    NRMSE=[]
    R = []
    FHV = []
    FLV = []
    for it in range(forecast_len):
        pred = Pred[:, it].flatten()
        tru  = Tru[:, it].flatten()
        mask = ~np.isnan(tru) & ~np.isnan(pred)
        pred, tru = pred[mask], tru[mask]
        kge = Hydrostats.kge_2012(pred, tru)
        nse = Hydrostats.nse(pred, tru)
        nrmse = Hydrostats.nrmse_range(pred, tru)
        rho = Hydrostats.pearson_r(pred, tru)
        fhv, _, flv = fdc_metrics(pred, tru)
        if not allRes is None:
            ens  = allRes[mask, :, it]
            crps = EM.ens_crps(tru, ens)['crpsMean']
            print (f'kge {kge}, nse {nse}, crps {crps}, fhv {fhv}, flv {flv}')
            CRPS.append(crps)
        else:
            print (f'kge {kge}, nse {nse}, fhv {fhv}, flv {flv}')
        KGE.append(kge)
        NSE.append(nse)
        NRMSE.append(nrmse)
        R.append(rho)
        FHV.append(fhv)
        FLV.append(flv)

    KGE = np.stack(KGE,axis=0)    
    NSE = np.stack(NSE,axis=0)
    if CRPS:
        CRPS = np.stack(CRPS,axis=0)
    NRMSE = np.stack(NRMSE,axis=0)
    R = np.stack(R, axis=0)
    FHV = np.stack(FHV)
    FLV = np.stack(FLV)
    metric_dict = {'kge': KGE, 'nse': NSE, 'crps': CRPS, 'nrmse':NRMSE, 'R': R,
                   'fhv': FHV, 'flv': FLV
                    }
    return metric_dict

def zeroshot(
        cfg,
        model, 
        model_type: str,
        col_name: str, 
        df: pd.DataFrame, 
        lookback: int = 32, 
        forecast_len: int = 1,
        dct_stat: Dict = None
    ):
    """
    model_type, 'sundial', 'chronos'
    col_name, key of the output dictionary, used as gage_id in my case
    arr: numpy array of the inputs
    lookback, length of lookback window
    forecast_len, length of forecast window
    """
    
    dataset = formDataSet(gageid=col_name,  
                          df_raw=df, 
                          lookback=lookback, 
                          forecast_len=forecast_len,
                          label_len=0, 
                          allow_missing=cfg.model.allow_missing,
                )
    
    #calculate full dataset size
    d0 = pd.to_datetime(cfg.model.start_date)
    d1 = pd.to_datetime(cfg.model.end_date)
    #total number of 3H points
    num_data = int((d1 - d0).days*24/3) - lookback - forecast_len + 1

    #remove bad stations for zero-shot learning
    if dataset is None:
        warnings.warn(f"gage {col_name} returns empty dataset")
        return None
    elif len(dataset)/num_data < cfg.model.min_data_fraction:
        msg = f"gage {col_name} has insufficient data, {len(dataset)} {num_data}"
        warnings.warn(msg)
        return None

    dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            drop_last= False
    )                

    # prepare input
    num_samples = 10

    Tru = []
    Pred = []
    allRes = []
    Q10 = []
    Q90 = []

    for x, y,_,_ in tqdm(dataloader):
        #[batch, lookback]
        if len(x.shape)>2:
            x = x.squeeze(-1)        
        seqs = x.to(device)
        if model_type == 'sundial':
            # Note that Sundial can generate multiple probable predictions
            #forecast shape, [batch, nsamples, forecast_len]
            forecast = model.generate(seqs, max_new_tokens=forecast_len, 
                                      num_samples=num_samples).data.cpu()
            allRes.append(forecast)
            mu = forecast.mean(dim=1).numpy()
            Pred.append(mu)
            Tru.append(y.data.cpu().numpy())
            Q10.append(forecast[...,0].quantile(q=0.1, dim=0).data.cpu().numpy())
            Q90.append(forecast[...,0].quantile(q=0.9, dim=0).data.cpu().numpy())

        elif model_type == 'chronos':
            forecast, mu = model.predict_quantiles(
                seqs,
                prediction_length=forecast_len, 
                quantile_levels=[0.1, 0.9],
                limit_prediction_length = True,
            )
            Pred.append(mu.data.cpu().numpy())
            Tru.append(y.data.cpu().numpy())
            Q10.append(forecast[...,0].data.cpu().numpy())
            Q90.append(forecast[...,1].data.cpu().numpy())
        elif model_type == 'moirai':
            dataset = ListDataset(
                data_iter=[
                    {
                        "start": '2005-10-01',
                        "target": x[i].data.cpu().numpy(),
                        "feat_static_cat": [0],  # dummy category
                    }
                    for i in range(x.shape[0])],
                freq="3H"  # or whatever frequency your time series has
            )
            predictor = model.create_predictor(batch_size=x.shape[0])
            forecasts = list(predictor.predict(dataset))
            quantile_levels = [0.1, 0.5, 0.9]  # e.g., 10th, 50th (median), and 90th percentiles

            samples = np.stack([forecasts[i].samples for i in range(len(forecasts))], axis=0)

            quantiles = {
                f"q{int(q * 100)}": np.quantile(samples, q=q, axis=1)
                for q in quantile_levels
            }
                        
            mu  = quantiles['q50']
            q10 = quantiles['q10']
            q90 = quantiles['q90']
            Pred.append(mu)
            Tru.append(y.data.cpu().numpy())
            allRes.append(samples)

    Pred = np.concatenate(Pred, axis=0)
    Tru = np.concatenate(Tru, axis=0)
    if cfg.model.normalization:
        Pred = Pred*dct_stat['qstd'] + dct_stat['qmean']
        Tru  = Tru*dct_stat['qstd'] + dct_stat['qmean']

    if cfg.model.log_transform:
        if cfg.model.transform_method == 'log':
            Pred = np.exp(Pred)-1
            Tru  = np.exp(Tru)-1
        elif cfg.model.transform_method == 'feng':
            Pred = np.square(10**(Pred) - 0.1)  #np.log10(np.sqrt(arr) + 0.1)
            Tru =  np.square(10**(Tru)  - 0.1)
        else:
            raise ValueError("invalid transform method")        
    
    if allRes:
        allRes = np.concatenate(allRes, axis=0)
        if cfg.model.normalization:
            allRes = allRes*dct_stat['qstd'] + dct_stat['qmean']
        if cfg.model.log_transform:
            if cfg.model.transform_method == 'log':
                allRes = np.exp(allRes)-1                
            elif cfg.model.transform_method == 'feng':
                allRes = np.square(10**(allRes) - 0.1)  #np.log10(np.sqrt(arr) + 0.1)
            else:
                raise ValueError("invalid transform method")        
    else:
        allRes = None
    
    metric_dict = getMetrics(Pred, Tru, forecast_len, allRes)
    return metric_dict

def ttm_zeroshot(cfg, tsp, pipeline, gage_df, dct_stat):
    """Zeroshot module for IBM TTM
    Params:
    ------
    tsp: TimeSeriesPreprocessor
    pipeline: TimeSeriesForecastingPipeline
    gage_df: dataframe containing streamflow data
    dct_stat: dictionary of data stats
    
    Returns
    -------
    metric_dict, dictionary of metrics
    """
    gage_df['timestamp'] = pd.to_datetime(gage_df.index)    
    gage_df['item_id']   = 1   #hardcode to 1 because we handle 1 TS per call
    gage_df["Year"] = gage_df.timestamp.dt.year
    gage_df["Mnth"] = gage_df.timestamp.dt.month
    gage_df["Day"]  = gage_df.timestamp.dt.day

    zeroshot_forecast = pipeline(tsp.preprocess(gage_df))    
    
    Tru  = np.vstack(zeroshot_forecast['Q'].values)
    Pred = np.vstack(zeroshot_forecast['Q_prediction'].values)
   
    if cfg.model.normalization:
        Pred = Pred*dct_stat['qstd'] + dct_stat['qmean']
        Tru  =  Tru*dct_stat['qstd'] + dct_stat['qmean']

    if cfg.model.log_transform:
        if cfg.model.transform_method == 'log':
            Pred = np.exp(Pred)-1
            Tru  = np.exp(Tru)-1
        elif cfg.model.transform_method == 'feng':
            Pred = np.square(10**(Pred) - 0.1)  #np.log10(np.sqrt(arr) + 0.1)
            Tru =  np.square(10**(Tru)  - 0.1)
        else:
            raise ValueError("invalid transform method")        

    
    metric_dict = getMetrics(Pred, Tru, forecast_len=cfg.model.pred_len, allRes=None)    

    return metric_dict


def postprocessing(cfg, outdir, model_type):
    """Postprocess results """
    print ('*'*10, 'Model used is ', model_type)

    NSE = []
    KGE = []
    R = []
    FHV=[]
    FLV=[]
    if cfg.model.log_transform:
        if not cfg.model.allow_missing:
            metrics = pkl.load(open(f'{outdir}/metrics_log_{cfg.model.transform_method}.pkl', 'rb'))
        else:
            metrics = pkl.load(open(f'{outdir}/metrics_allow_missing_log_{cfg.model.transform_method}.pkl', 'rb'))            
    else:
        if not cfg.model.allow_missing:
            metrics = pkl.load(open(f'{outdir}/metrics.pkl', 'rb'))
        else:
            metrics = pkl.load(open(f'{outdir}/metrics_allow_missing.pkl', 'rb'))
    
    #load bad stations [hard code for 365/1 use only]
    if not cfg.model.allow_missing:
        badStations = pkl.load(open(f'bad_camels_3h_zeroshot_stations{cfg.model.seq_len}_{cfg.model.pred_len}.pkl', 'rb'))
    else:
        badStations = pkl.load(open(f'bad_camels_3h_zeroshot_stations_allow_missing_{cfg.model.seq_len}_{cfg.model.pred_len}.pkl', 'rb'))

    for item in metrics.keys():        
        if item not in badStations:
            NSE.append(metrics[item]['nse'])
            KGE.append(metrics[item]['kge'])
            R.append(metrics[item]['R'])
            FHV.append(metrics[item]['fhv'])
            FLV.append(metrics[item]['flv'])
            #print (item, metrics[item]['nse'][0])
    NSE = np.stack(NSE)
    KGE = np.stack(KGE)
    R   = np.stack(R)
    FHV = np.stack(FHV)
    FLV = np.stack(FLV)
    
    for it in range(cfg.model.pred_len):
        print (it, 'Median NSE', np.median(NSE[:,it]), 'Mean NSE', np.mean(NSE[:,it]))
        print (it, 'Median KGE', np.median(KGE[:,it]), 'Mean KGE', np.mean(KGE[:,it]))
        print (it, 'Median R', np.median(R[:,it]), 'Mean R', np.mean(R[:,it]))
        print (it, 'Median FHV', np.median(FHV[:,it]), 'Mean FHV', np.mean(FHV[:,it]))
        print (it, 'Median FLV', np.median(FLV[:,it]), 'Mean FLV', np.mean(FLV[:,it]))
    print ('NSE shape', NSE.shape)
    return NSE, KGE, R, FHV, FLV

def gen_df(cfg, gage_id:int, mode:str):
    """Generate dataframe for TSFM to use
    Params:
    ------
    gage_id: usgs gauge id 
    mode: 'train', 'test'

    Return:
    ------
    gageDF: dataframe containing both forcing and streamflow data
    """

    _, basin_area = load_forcing(cfg, forcingType='nldas', basin=gage_id, mode=mode)
    gageDF = getUSGSData(cfg, gage_id, mode=mode, area=basin_area)
    
    #mask invalid data
    if not gageDF is None:
        arr = gageDF.Q.values
        arr[arr<=0] = np.nan #use this line for 56-8 case used in Table 2.
        arr[arr<0] = np.nan
        if cfg.model.log_transform:
            if cfg.model.transform_method=='log':
                gageDF.Q = np.log1p(arr)
            elif cfg.model.transform_method=='feng':                    
                gageDF.Q = np.log10(np.sqrt(arr) + 0.1)
            else:
                raise ValueError("Invalid transform method")                         
        else:
            gageDF.Q = arr

        gageDF['timestamp'] = pd.to_datetime(gageDF.index)
        gageDF['item_id'] = gage_id
    return gageDF

def main(regen=False):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse arguments for training CAMELS",
    )

    parser.add_argument('--config', type=str, required=True, help='config file name')

    cargs = parser.parse_args()
    config_file = cargs.config
    cfg = OmegaConf.load(open(config_file, "r"))
    config_model = cfg.model
    basinDF = loadCAMELS_List(cfg)
    basinlist = basinDF['gage_id'].to_list()
    
    model_type = config_model.model_type

    conda_env_name = os.environ.get('CONDA_DEFAULT_ENV')
    
    if model_type == 'sundial':
        assert(conda_env_name == 'tslm')
        model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True).to(device)
        model.config.use_cache = False
    elif model_type == 'chronos':
        assert(conda_env_name== 'chronos')
        model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            torch_dtype=torch.bfloat16,
            device_map= device
        )        
    elif model_type == 'moirai':
        assert(conda_env_name == 'moirai')
        
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-base"),
            prediction_length=cfg.model.pred_len,
            context_length=cfg.model.seq_len,
            patch_size="auto",
            num_samples=cfg.model.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(device)

    elif model_type == 'ttm':
        assert(conda_env_name == 'ibm')
        
        model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2",  # Name of the model on Hugging Face
            num_input_channels=1,  # tsp.num_input_channels
            prediction_filter_length=cfg.model.pred_len,
        )
        #to use ttm, the bad station list must already exist
        badstation_file = f"bad_camels_3h_zeroshot_stations{cfg.model.seq_len}_{cfg.model.pred_len}.pkl"
        assert(os.path.exists(badstation_file))
        existingbadStations = pkl.load(open(badstation_file, 'rb'))
    else:
        raise Exception("Invalid model type")

    #in 3-hr intervals
    lookback_len = config_model.seq_len
    forecast_len = config_model.pred_len        
    out_dir = f'camels3h{model_type}_{lookback_len}_{forecast_len}'
    os.makedirs(out_dir, exist_ok=True) 

    if regen: 
        counter = 0
        #gather test dataframes
        input_df = []
        for gage_id in tqdm(basinlist):
            gageDF = gen_df(cfg, gage_id, mode="test")
            if not gageDF is None:
                input_df.append(gageDF)
        
        dct = None
        if config_model.normalization:    
            statsfile = 'camels_3h_stats.pkl'                 
            if not os.path.exists(statsfile):
                #need to regenerate stats based on train data
                train_df = []
                for gage_id in tqdm(basinlist):
                    gageDF = gen_df(cfg, gage_id, mode="train")
                    if not gageDF is None:
                        train_df.append(gageDF)
                train_df = pd.concat(train_df)
                qarr = train_df['Q'].values      
                q_mean = np.nanmean(qarr)
                q_std = np.nanstd(qarr)
                pkl.dump({'qmean':q_mean, 'qstd':q_std}, open(statsfile, 'wb'))
                del train_df
             
            dct = pkl.load(open(statsfile, 'rb'))
            q_mean = dct['qmean']
            q_std  = dct['qstd']
            for df in input_df:
                arr = (df.Q - q_mean)/q_std
                df.Q = arr

        counter=0
        badStations=[]
        metrics = {}
        for gageDF in tqdm(input_df):
            if gageDF is None or gageDF.empty:
                print (f'skipping {gage_id}')
                continue
            print (f'Processing {counter} {gage_id}')
            counter+=1
            gage_id = gageDF['item_id'].values[0]

            if model_type in ['sundial', 'chronos', 'moirai']:
                res = zeroshot(
                    cfg,
                    model, 
                    model_type,        
                    col_name=gage_id, 
                    df = gageDF, 
                    lookback=lookback_len, 
                    forecast_len=forecast_len,
                    dct_stat=dct
                )
                if not res is None:
                    metrics[gage_id] = res
                else:
                    print ('skipped ', gage_id)
                    badStations.append(gage_id.item())

            elif model_type == 'ttm': 
                if gage_id in existingbadStations:
                    print ('skip gage_id')
                    continue
                seq_len  = cfg.model.seq_len
                pred_len = cfg.model.pred_len
                
                target_columns = ["Q"]
                control_columns= ["Year","Mnth", "Day"]
                timestamp_column = "timestamp"
                id_columns = ["item_id"]

                tsp = TimeSeriesPreprocessor(
                    timestamp_column=timestamp_column,
                    target_columns=target_columns,
                    id_columns= id_columns,
                    control_columns=control_columns, #if this is removed, "Expecting freq_token in forward" exception will be raised
                    context_length = seq_len,
                    prediction_length=pred_len,
                    encode_categorical=False,
                    scaling=False,  #this should be False
                    scaler_type="standard",
                )

                pipeline = TimeSeriesForecastingPipeline(
                    model,
                    timestamp_column=timestamp_column,
                    id_columns=id_columns,
                    target_columns=target_columns,
                    control_columns = control_columns,
                    explode_forecasts=False,
                    freq="3H",
                    device=device,  # Specify your local GPU or CPU.
                    batch_size = cfg.batch_size,
                )

                metrics[gage_id] = ttm_zeroshot(
                    cfg,
                    tsp, 
                    pipeline,
                    gage_df = gageDF,
                    dct_stat = dct
                )

            else:
                raise Exception('bad model type')
        if model_type in ['sundial', 'chronos', 'moirai']:
            print (f'Number of bad stations {len(badStations)}')
            if not cfg.model.allow_missing:
                pkl.dump(badStations, open(f"bad_camels_3h_zeroshot_stations{cfg.model.seq_len}_{cfg.model.pred_len}.pkl", "wb"))
            else:
                pkl.dump(badStations, open(f"bad_camels_3h_zeroshot_stations_allow_missing_{cfg.model.seq_len}_{cfg.model.pred_len}.pkl", "wb"))
        if cfg.model.log_transform:
            if not cfg.model.allow_missing:
                pkl.dump(metrics, open(f'{out_dir}/metrics_log_{cfg.model.transform_method}.pkl', 'wb'))
            else:
                pkl.dump(metrics, open(f'{out_dir}/metrics_allow_missing_log_{cfg.model.transform_method}.pkl', 'wb'))
        else:
            if not cfg.model.allow_missing:            
                pkl.dump(metrics, open(f'{out_dir}/metrics.pkl', 'wb'))            
            else:
                pkl.dump(metrics, open(f'{out_dir}/metrics_allow_missing.pkl', 'wb'))            
    else:
        postprocessing(cfg, out_dir, cfg.model.model_type)

if __name__ == '__main__':
    main(regen=False)
