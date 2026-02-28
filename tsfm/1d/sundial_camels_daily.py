#author: alex sun
#>python sundial_camels_daily.py --config config_sundial_camelsdaily.yaml
#date: 06/03/2025
#rev date: 08/16/2025
#rev date: 08/28/2025, changed to new camels_dataloader.py
#Test zero-shot on camels
#conda env: tslm for sundial
#conda env: chronos for chronos
#conda env: moiroi for moirai
#command to run
#>python sundial_camels.py --config config_sundial_camels.yaml
#rev date: 09222025, 
# -revised data normalization routines
#rev date: 10072025, cleanup for zeroshot, add normalization
#rev date: 10192025, add sample counter for missing data
#rev date: 11042025, this was used Table 2, Figure 2
#rev date: 01232026, paper revision
#=======================================================================
import os,sys
import warnings
from typing import List,Dict, Any
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

from camels_dataloader_daily import formDataSet,loadCAMELS_List,getUSGSData,load_forcing
from camels_utils import fdc_metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ('='*10, 'Using device ', device)

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
            ens  = allRes[:, :, it]
            crps = EM.ens_crps(tru, ens)['crpsMean']
            print (f'kge {kge:5.3f}, nse {nse:5.3f}, crps {crps:5.3f}, fhv {fhv:6.3f}, flv {flv:6.3f}')
            CRPS.append(crps)
        else:
            print (f'kge {kge:5.3f}, nse {nse:5.3f}, fhv {fhv:6.3f}, flv {flv:6.3f}')
        print (f'{kge:5.3f} & {nse:5.3f} & {fhv:6.3f} & {flv:6.3f} \\\\')
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

def count_samples(
       cfg,
       col_name: str, 
       df: pd.DataFrame, 
       lookback: int = 32, 
       forecast_len: int = 1,       
    ):
    dataset = formDataSet(gageid=col_name,  df_raw= df, 
                          lookback=lookback, forecast_len=forecast_len,
                          label_len=0, include_forcing=False)
    #calculate full dataset size
    d0 = pd.to_datetime(cfg.model.start_date)
    d1 = pd.to_datetime(cfg.model.end_date)
    #total number of possible data pairs
    num_days = (d1 - d0).days - lookback - forecast_len + 1
    if not dataset is None:
        print (len(dataset), num_days)
        return (len(dataset),num_days)
    else:
        return None

def zeroshot(
        cfg,
        model: Any,  
        model_type: str,
        col_name: str, 
        df: pd.DataFrame, 
        lookback: int = 32, 
        forecast_len: int = 1,
        dct_stat: Dict = None
    ):
    """
    Params:
    ------
    model_type, type of model
    col_name, used as gage_id
    df: dataframe of entire time series
    lookback: length of lookback window
    forecast_len, length of forecast horizon
    dct_stat: [Optional] dictionary of stats for normalize/denormalize
    """
    dataset = formDataSet(gageid=col_name,  df_raw= df, 
                          lookback=lookback, forecast_len=forecast_len,
                          label_len=0, 
                          include_forcing=False,
                          allow_missing=cfg.model.allow_missing)
    #calculate full dataset size
    d0 = pd.to_datetime(cfg.model.start_date)
    d1 = pd.to_datetime(cfg.model.end_date)
    num_days = float((d1 - d0).days) - lookback + 1
    
    #remove bad stations for zero-shot learning
    if dataset is None:
        warnings.warn(f"gage {col_name} returns empty dataset")
        return None    
    elif len(dataset)/num_days < cfg.model.min_data_fraction:
        print (len(dataset), num_days, len(dataset)/num_days)
        msg = f"gage {col_name} has insufficient data, num of data is {len(dataset)}"
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
    num_samples = cfg.model.num_samples

    #prepare output arrays
    Tru = []
    Pred = []
    allRes = []
    Q10 = []
    Q90 = []

    for x, y,_,_ in tqdm(dataloader):
        #x: [batch, lookback]
        if len(x.shape)>2:
            x = x.squeeze(-1)
        seqs = x.to(device)
        
        if model_type == 'sundial':
            # Note that Sundial can generate multiple probable predictions
            # forecast shape, [batch, nsamples, forecast_len]
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
                        "start": '1995-04-15',  #dummy start date
                        "target": x[i].data.cpu().numpy(),
                        "feat_static_cat": [0],  # dummy category
                    }
                    for i in range(x.shape[0])],
                freq="D"  # daily freq
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
    Tru  = np.concatenate(Tru, axis=0)
    
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
                allRes = (np.square(10**(allRes) - 0.1))
            else:
                raise ValueError("invalid transform method")
    else:
        allRes = None

    if cfg.model.allow_missing:
        allRes = None
    metric_dict = getMetrics(Pred, Tru, forecast_len, allRes)

    #as01232026, save for postprocessing
    if col_name in [6409000, 4122200, 14301000, 8324000, 2198100]:
        print ('saving output for ', col_name)
        pkl.dump([Pred, Tru], open(f'./paper_data/zeroshot/{model_type}_{col_name}.pkl', 'wb'))

    return metric_dict

def ttm_zeroshot(cfg, tsp, pipeline, gage_df, dct_stat, forecast_len, gage_id=None):
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

    zeroshot_forecast = pipeline(tsp.preprocess(gage_df))    

    Tru  = np.vstack(zeroshot_forecast['Q'].values)
    Pred = np.vstack(zeroshot_forecast['Q_prediction'].values)

    print ('Pred shape', Pred.shape)
    
    
    if cfg.model.normalization:
        Pred = Pred*dct_stat['qstd'] + dct_stat['qmean']
        Tru  = Tru*dct_stat['qstd'] + dct_stat['qmean']

    if cfg.model.log_transform:
        if cfg.model.transform_method == 'log':
            Pred = np.exp(Pred)-1
            Tru  = np.exp(Tru)-1
        elif cfg.model.transform_method == 'feng':
            Pred = np.square(10**Pred - 0.1)
            Tru =  np.square(10**Tru  - 0.1)
        else:
            raise ValueError("invalid transform method")
    
    metric_dict = getMetrics(Pred, Tru, forecast_len=forecast_len, allRes=None)    

    #as01232026, save for postprocessing
    print ('saving output for ', gage_id)
    pkl.dump([Pred, Tru], open(f'./paper_data/zeroshot/ttm_{gage_id}.pkl', 'wb'))


    return metric_dict

def gen_df(cfg, gage_id:int, mode:str, return_basin_area:bool=False):
    """Generate dataframe for TSFM to use
    Params:
    ------
    gage_id: usgs gauge id 
    mode: 'train', 'test'
    return_basin_area: True to return basin drainage area

    Return:
    ------
    gageDF: dataframe containing both forcing and streamflow data
    """

    _, basin_area = load_forcing(cfg, forcingType='nldas', basin=gage_id, mode=mode)
    gageDF = getUSGSData(cfg, gage_id, mode=mode, area=basin_area)

    #mask invalid data
    arr = gageDF.Q.values
    #
    if cfg.model.remove_zero_flow:
        arr[arr<=0] = np.nan
    else:
        arr[arr<0] = np.nan

    if cfg.model.log_transform:
        if cfg.model.transform_method == 'log':
            gageDF.Q = np.log1p(arr)
        elif cfg.model.transform_method == 'feng':
            gageDF.Q = np.log10(np.sqrt(arr) + 0.1)
        else:
            raise ValueError("Invalid transform method")
    else:
        gageDF.Q = arr

    gageDF = gageDF.drop(['flag', 'basin'], axis=1)
    gageDF['timestamp'] = pd.to_datetime(gageDF.index)
    gageDF['item_id'] = gage_id
    if return_basin_area:
        return gageDF, basin_area
    else:
        return gageDF

def postprocessing(cfg, outdir:str):
    """Postprocessing save metrics data
    Params:
    cfg: yml configuration 
    outdir: folder where the metrics pkl files are located
    Returns:
    -------
    NSE, 
    KGE,
    """
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
        metrics = pkl.load(open(f'{outdir}/metrics.pkl', 'rb'))
    
    #load bad stations [hard code for 365/1 use only]
    if not cfg.model.allow_missing:
        badStations = pkl.load(open(f'bad_camels_daily_zeroshot_stations{cfg.model.seq_len}.pkl', 'rb'))
    else:
        badStations = pkl.load(open(f"bad_camels_daily_zeroshot_stations_allowmissing_{cfg.model.seq_len}.pkl", "rb"))

    #!!!! 10/31/2025, override allow_missing in the above
    #the intention is to plot metrics for the no low-flow cases
    if not cfg.model.remove_zero_flow:
        badStations = pkl.load(open('bad_camels_daily_zeroshot_stations365_for_nolowflow.pkl', 'rb'))
    else:
        badStations = []

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

    #print ('*'*30)
    #print ('nse shape', NSE.shape)
    
    for it in range(cfg.model.pred_len):
        print (it, 'Median NSE', np.median(NSE[:,it]), 'Mean NSE', np.mean(NSE[:,it]))
        print (it, 'Median KGE', np.median(KGE[:,it]), 'Mean KGE', np.mean(KGE[:,it]))
        print (it, 'Median R', np.median(R[:,it]), 'Mean R', np.mean(R[:,it]))
        print (it, 'Median FHV', np.median(FHV[:,it]), 'Mean FHV', np.mean(FHV[:,it]))
        print (it, 'Median FLV', np.median(FLV[:,it]), 'Mean FLV', np.mean(FLV[:,it]))
    return NSE, KGE, R, FHV, FLV

def main(regen:bool=False):
    """Main driver
    Params:
    regen: True to generate the metrics
    
    """
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
    
    #for testing only
    basinlist = [6409000, 4122200, 14301000, 8324000, 2198100]

    model_type = config_model.model_type
    print ('model type is ', model_type)

    conda_env_name = os.environ.get('CONDA_DEFAULT_ENV')

    #fix the random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

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
        
    else:
        raise Exception("Invalid model type")
    
    #in 1-d intervals
    lookback_len = config_model.seq_len
    forecast_len = config_model.pred_len    
    model_type   = config_model.model_type
    print ('*'*30)
    print ('Model type', model_type, 'lookback', lookback_len, 'forecast len', forecast_len)
    print ('*'*30)

    out_dir = f'camelsdaily_{model_type}_{lookback_len}_{forecast_len}'
    os.makedirs(out_dir, exist_ok=True) 
    metrics = {}

    if regen: 
        mode = 'test'
        input_df = []
        for gage_id in tqdm(basinlist):
            gageDF = gen_df(cfg, gage_id, mode)
            input_df.append(gageDF)

        #asun: 11072025, uncomment the following to get all camels basin areas    
        #basin_areas = {}
        # code for generating basin area pkl file
        # for gage_id in tqdm(basinlist):
        #     gageDF, basin_area = gen_df(cfg, gage_id, mode, return_basin_area=True)
        #     print (basin_area)
        #     basin_areas[gage_id] = basin_area
        #     input_df.append(gageDF)
        # pkl.dump(basin_areas, open('camels_basin_areas.pkl', 'wb'))

        dct = None
        if config_model.normalization:                        
            stats_file = 'zeroshot_camels_daily_stats'
            if cfg.model.log_transform:
                stats_file += f'_{cfg.model.transform_method}'
            stats_file +='.pkl'
            if not os.path.exists(stats_file):
                #need to regen stats
                print ('!'*10, 're-generate statistics')
                train_df = []
                for gage_id in tqdm(basinlist):
                    gageDF = gen_df(cfg, gage_id, "train")
                    train_df.append(gageDF)
                train_df = pd.concat(train_df)
                qarr = train_df['Q'].values      
                q_mean = np.nanmean(qarr)
                q_std = np.nanstd(qarr)
                pkl.dump({'qmean':q_mean, 'qstd':q_std}, open(stats_file, 'wb'))
                del train_df
            dct = pkl.load(open(stats_file, 'rb'))
            q_mean = dct['qmean']
            q_std  = dct['qstd']
        
            for df in input_df:
                arr = (df.Q - q_mean)/q_std
                df.Q = arr

        counter=0
        badStations=[]

        #missing data ratio =   0.07823157953954119
        reCalMissingData = False #[this is not used]
        if reCalMissingData:
            #10/19/2025, count number of missing samples for the paper
            totalSamples = 0
            actualSamples = 0
            basin_missing = {}
            for gageDF in tqdm(input_df):
                gage_id = gageDF['item_id'].values[0]
                res =count_samples(cfg, 
                            col_name=gage_id, 
                            df = gageDF, 
                            lookback=lookback_len, 
                            forecast_len=forecast_len)
                if not res is None:
                    totalSamples+=res[1]
                    actualSamples+=res[0]
                    basin_missing[gage_id]= res[0]/res[1]
            print ('Missing sample percentage', 1-actualSamples/totalSamples)
            pkl.dump(basin_missing, open(f'missingdata_fractions_{lookback_len}_{forecast_len}.pkl', 'wb'))

        for gageDF in tqdm(input_df):
            gage_id = gageDF['item_id'].values[0]
            if gageDF is None or gageDF.empty:
                warnings.warn(f'Skipping {gage_id}')
                continue
            #print (f'Processing {counter} {gage_id}')
            counter+=1
            if model_type in ['sundial', 'chronos', 'moirai']:
                res = zeroshot(
                    cfg,
                    model, 
                    model_type,        
                    col_name=gage_id, 
                    df = gageDF, 
                    lookback=lookback_len, 
                    forecast_len=forecast_len,
                    dct_stat = dct
                )
                if not res is None:
                    metrics[gage_id] = res
                else:
                    print ('skipped ', gage_id)
                    badStations.append(gage_id.item())

            elif model_type == 'ttm':
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
                    freq="1D",
                    device=device,  # Specify your local GPU or CPU.
                    batch_size = cfg.batch_size,
                )

                metrics[gage_id] = ttm_zeroshot(
                    cfg,
                    tsp, 
                    pipeline,
                    gage_df = gageDF,
                    dct_stat = dct,
                    forecast_len = pred_len,
                    gage_id=gage_id
                )
        return
        if model_type in ['sundial', 'chronos', 'moirai']:
            if not cfg.model.allow_missing:
                pkl.dump(badStations, open(f"bad_camels_daily_zeroshot_stations{cfg.model.seq_len}.pkl", "wb"))
            else:
                pkl.dump(badStations, open(f"bad_camels_daily_zeroshot_stations_allowmissing_{cfg.model.seq_len}.pkl", "wb"))

        if cfg.model.log_transform:
            if not cfg.model.allow_missing:
                pkl.dump(metrics, open(f'{out_dir}/metrics_log_{cfg.model.transform_method}.pkl', 'wb'))
            else:
                pkl.dump(metrics, open(f'{out_dir}/metrics_allow_missing_log_{cfg.model.transform_method}.pkl', 'wb'))
        else:
            pkl.dump(metrics, open(f'{out_dir}/metrics.pkl', 'wb'))
    else:
        postprocessing(cfg, out_dir)

if __name__ == '__main__':
    main(regen=False)
