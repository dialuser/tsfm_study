#author: alex sun
#date: 08/18/2025
#purpose: train a base case for 3-hr prediction
#conda: conda activate mesh
#>python lstm_main.py --config config_lstm_camels.yaml
#11/04/2025, this was used to generate 3H LSTM results for the paper
#====================================================================
import math
import shutil
import os,sys
import numpy as np
import pickle as pkl
from datetime import timedelta

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from omegaconf import OmegaConf

from lstm import MyLSTM
from camels_dataloader import AllDataloader
import hydrostats as Hydrostats
from camels_utils import fdc_metrics

logger = get_logger(__name__, log_level="INFO")

def postprocess(cfg, seed=None, badstations=None):
    if seed is None:
        seed = cfg.seed
    metrics = pkl.load(open(f'camels_lstm_{cfg.model.seq_len}_{cfg.model.pred_len}.pkl', 'rb'))

    NSE = []
    KGE = []
    R   = []
    FHV = []
    FLV = []

    for item in metrics.keys():
        if not badstations is None and item in badstations:
            continue
        NSE.append(metrics[item]['nse'])
        KGE.append(metrics[item]['kge'])
        R.append(metrics[item]['rho'])
        FHV.append(metrics[item]['fhv'])
        FLV.append(metrics[item]['flv'])

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
        print (it, 'Median R',   np.median(R[:,it]), 'Mean R', np.mean(R[:,it]))
        print (it, 'Median FHV', np.median(FHV[:,it]), 'Mean FHV', np.mean(FHV[:,it]))
        print (it, 'Median FLV', np.median(FLV[:,it]), 'Mean FLV', np.mean(FLV[:,it]))                
    #print latex str
    print(f""" \
        {np.median(NSE[:,0]):4.3f}/{np.mean(NSE[:,0]):4.3f} &  \
        {np.median(KGE[:,0]):4.3f}/{np.mean(KGE[:,0]):4.3f} &  \
        {np.median(FHV[:,0]):4.3f}/{np.mean(FHV[:,0]):4.3f} &  \
        {np.median(FLV[:,0]):4.3f}/{np.mean(FLV[:,0]):4.3f} \\\\   
        {np.median(NSE[:,-1]):4.3f}/{np.mean(NSE[:,-1]):4.3f} &  \
        {np.median(KGE[:,-1]):4.3f}/{np.mean(KGE[:,-1]):4.3f} &  \
        {np.median(FHV[:,-1]):4.3f}/{np.mean(FHV[:,-1]):4.3f} &  \
        {np.median(FLV[:,-1]):4.3f}/{np.mean(FLV[:,-1]):4.3f} \\\\         
          """)
    return (NSE, KGE, FHV, FLV)

def eval(model, dataloader, loss_criterion):
    model.eval()
    total_loss = []
    progress_bar = tqdm(total=len(dataloader)) #, disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Validation")

    N = 0
    for step, (X, target, _, _) in enumerate(dataloader):           
        with torch.no_grad():
            model_output =  model(X)

            if torch.isnan(model_output).any():
                raise Exception('null cells found')
            loss =loss_criterion(model_output, target)  # this could have different weights!

            total_loss.append(loss.item())
            N += X.shape[0]
    total_loss = np.sum(total_loss)/N  #loss per data
    return total_loss

def main(reTrain=False, regen=False):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse arguments for lstm",
    )

    #Only used for inductive experiments
    parser.add_argument('--config',   type=str, required=True, help='config file name')

    cargs = parser.parse_args()
    cfg = OmegaConf.load(open(cargs.config, "r"))
    
    config_model = cfg.model

    #Fix the seed for reproducibility
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    config_model.output_dir += f'{config_model.seq_len}_{config_model.pred_len}'
    os.makedirs(config_model.output_dir, exist_ok=True)
    logging_dir = os.path.join(config_model.output_dir, config_model.logging_dir)

    model = MyLSTM(
        input_size = config_model.in_dim,
        hidden_size = config_model.hidden_size,
        pred_len = config_model.pred_len, 
        batch_first = True,
        num_layers= config_model.num_layers,
        #initial_forget_bias =0
        )
    
    accelerator_project_config = ProjectConfiguration(project_dir=config_model.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=config_model.gradient_accumulation_steps,
        mixed_precision=config_model.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    accelerator.print(f"ACCELERATOR DEVICE:{accelerator.distributed_type}---- NUM OF PROCESSES: {accelerator.num_processes }")

    weight_dtype = torch.float32
    
    #load data
    allDataLoader   = AllDataloader(cfg)
    trainDataloader = allDataLoader.getDataLoader(mode='train', reload=False, regen_stats=False)
    valDataloader   = allDataLoader.getDataLoader(mode='val', reload=False, regen_stats=False)

    device = accelerator.device

    #============== Training starts there! ================
    if reTrain:
        #asun: these are in terms of number of batches
        num_warmup_steps_for_scheduler = config_model.lr_warmup_steps*accelerator.num_processes
        len_train_per_proc = math.ceil(len(trainDataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_per_proc / config_model.gradient_accumulation_steps)
        num_training_steps_for_scheduler = config_model.num_epochs * num_update_steps_per_epoch * accelerator.num_processes
        max_train_steps = num_update_steps_per_epoch *  num_update_steps_per_epoch 

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config_model.learning_rate,
            betas=(config_model.adam_beta1, config_model.adam_beta2),
            weight_decay=float(config_model.adam_weight_decay),
            eps=float(config_model.adam_epsilon),
        )

        #create a learning rate scheduler
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler
        )
  
        model, optimizer, lr_scheduler, trainDataloader,valDataloader = accelerator.prepare(model, optimizer, lr_scheduler, trainDataloader, valDataloader)
        print ('train len', len(trainDataloader))

        if config_model.use_ema:
            #this is needed to model model_config work
            model = accelerator.unwrap_model(model)

            ema_model = EMAModel(
                model.parameters(),
                decay=config_model.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=config_model.ema_inv_gamma,
                power=config_model.ema_power,
                model_cls=MyLSTM,
                model_config=model.config,
            )

            ema_model.to(device)

        global_step = 0
        first_epoch = 0        

        if cfg.model.norm=='l1':
            loss_criterion = nn.L1Loss()
        elif cfg.model.norm=='l2':
            loss_criterion = nn.MSELoss()
        else:
            raise ValueError("invalid loss fun option")       
        
        best_val = np.inf
        for epoch in range(first_epoch, first_epoch+config_model.num_epochs):
            model.train()
            progress_bar = tqdm(total=num_update_steps_per_epoch) #, disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            train_loss = 0.0

            for step, (X,target,_,_) in enumerate(trainDataloader):
                target = target.squeeze(-1)
            
                #make the channel dimension the last dimension
                with accelerator.accumulate(model):
                    X = X.to(weight_dtype)
                    target = target.to(weight_dtype)
                    model_output = model(X).squeeze(-1)

                    if torch.isnan(model_output).any():
                        raise Exception('null cells found')
                    loss =loss_criterion(model_output, target)  # this could have different weights!

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(config_model.batch_size)).mean()
                    train_loss += avg_loss.item() / config_model.gradient_accumulation_steps

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        if config_model.use_ema:
                            ema_model.step(model.parameters())
                        progress_bar.update(1)
                        global_step += 1

                        if accelerator.is_main_process:
                            if global_step % config_model.checkpointing_steps == 0:
                                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                                if config_model.checkpoints_total_limit is not None:
                                    checkpoints = os.listdir(config_model.output_dir)
                                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                    if len(checkpoints) >= config_model.checkpoints_total_limit:
                                        num_to_remove = len(checkpoints) - config_model.checkpoints_total_limit + 1
                                        removing_checkpoints = checkpoints[0:num_to_remove]

                                        logger.info(
                                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                        )
                                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                        for removing_checkpoint in removing_checkpoints:
                                            removing_checkpoint = os.path.join(config_model.output_dir, removing_checkpoint)
                                            shutil.rmtree(removing_checkpoint)

                                #save_path = os.path.join(config_model.output_dir, f"checkpoint-{global_step}")
                                #accelerator.save_state(save_path, safe_serialization=False)
                                #os.makedirs(save_path, exist_ok=True)
                                #torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
                                #logger.info(f"Saved state to {save_path}")

                    logs = {"loss": loss.detach().item(), 'trainloss': train_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                    if config_model.use_ema:
                        logs["ema_decay"] = ema_model.cur_decay_value
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)            

            progress_bar.close()
            if accelerator.is_main_process:
                val_loss = eval(model, valDataloader, loss_criterion)
                print ('epoch loss', train_loss/(max_train_steps), 'val loss', val_loss)
                if val_loss<best_val:
                    best_val = val_loss
                    print ('saving the best model')
                    torch.save(model.state_dict(), os.path.join(config_model.output_dir, f'bestmodel{cfg.seed}.pth'))
                    bad_counter = 0
                else:
                    bad_counter+=1
                    print (f'Patience {bad_counter} out of {config_model.patience}')
                    if bad_counter>= config_model.patience:
                        break
            
            accelerator.wait_for_everyone()        
        
        if accelerator.is_main_process:
            save_path = os.path.join(config_model.output_dir, f'final{cfg.seed}.pth')
            accelerator.save_state(save_path, safe_serialization=False)
    else:
        if regen:
            testDataloader = allDataLoader.getDataLoader(mode='test', reload=True)
            model, testDataloader = accelerator.prepare(model, testDataloader)
            if accelerator.is_main_process:            
                accelerator.print(f"Loading from best model", os.path.join(config_model.output_dir, f'bestmodel{cfg.seed}.pth'))
                model.load_state_dict(torch.load(os.path.join(config_model.output_dir, f'bestmodel{cfg.seed}.pth')))

                model.eval()
                resDict={}
                truDict={}
                for (X, target, _, _, batch_gageid) in tqdm(testDataloader):
                    target = target.squeeze(-1)
                    #make the channel dimension the last dimension
                    with torch.no_grad():
                        X = X.to(weight_dtype)
                        target = target.data.cpu().numpy()
                        model_output = model(X).squeeze(-1)
                        batch_gageid = batch_gageid.data.cpu().numpy()
                        model_output = model_output.data.cpu().numpy()
                        for ix, item in enumerate(batch_gageid.flatten()): 
                            if not item in resDict.keys():
                                resDict[item] = []
                                truDict[item] = []
                            resDict[item].append (model_output[ix,:])
                            truDict[item].append(target[ix,:])

                print ('total number of gages to postproces', {len(resDict.keys())})

                stats_file = 'camels_3Hstats'
                if cfg.model.log_transform:
                    stats_file += f'_{cfg.model.transform_method}'
                stats_file +='.pkl'

                dct_stat = pkl.load(open(stats_file, 'rb'))
                metrics = {}

                for item in resDict.keys():
                    pred = np.stack(resDict[item])
                    y = np.stack(truDict[item])
                    metrics[item] = {'nse':[], 'kge':[], 'nrmse':[], 'rho': [], 'fhv': [], 'flv':[]}
                    if cfg.model.normalization:
                        pred = pred*dct_stat['qstd'] + dct_stat['qmean']
                        y  = y*dct_stat['qstd'] + dct_stat['qmean']

                    if cfg.model.log_transform:
                        if cfg.model.transform_method == 'log':
                            pred = np.exp(pred)-1
                            y  = np.exp(y)-1
                        elif cfg.model.transform_method == 'feng':
                            pred = np.square(10**(pred) - 0.1)  #np.log10(np.sqrt(arr) + 0.1)
                            y =  np.square(10**(y)  - 0.1)
                        else:
                            raise ValueError("invalid transform method")

                    print (f'{Hydrostats.nse(pred[:,0], y[:,0]):5.3f}, {Hydrostats.nse(pred[:,-1], y[:,-1]):5.3f}')
                    for it in range(config_model.pred_len):
                        metrics[item]['nse'].append(Hydrostats.nse(pred[:,it], y[:,it]))
                        metrics[item]['kge'].append(Hydrostats.kge_2012(pred[:,it], y[:,it]))
                        metrics[item]['nrmse'].append(Hydrostats.nrmse_range(pred[:,it], y[:,it]))
                        metrics[item]['rho'].append(Hydrostats.pearson_r(pred[:,it], y[:,it]))
                        fhv, _, flv = fdc_metrics(pred[:,it], y[:,it])
                        metrics[item]['fhv'].append(fhv)
                        metrics[item]['flv'].append(flv)

                    metrics[item]['nse'] = np.stack(metrics[item]['nse'], axis=0)
                    metrics[item]['kge'] = np.stack(metrics[item]['kge'], axis=0)
                    metrics[item]['nrmse'] = np.stack(metrics[item]['nrmse'], axis=0)
                    metrics[item]['rho'] = np.stack(metrics[item]['rho'], axis=0)
                    metrics[item]['fhv'] = np.stack(metrics[item]['fhv'], axis=0)
                    metrics[item]['flv'] = np.stack(metrics[item]['flv'], axis=0)

                pkl.dump(metrics, open(f'camels_lstm_{config_model.seq_len}_{config_model.pred_len}.pkl', 'wb'))
        else:
            badStations = pkl.load(open('bad_camels_3h_zeroshot_stations56_8.pkl', 'rb'))
            print (len(badStations))
            postprocess(cfg, badstations=badStations)

if __name__ == '__main__':    
    main(reTrain=False, regen=False)
