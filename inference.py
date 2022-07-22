import torch
from torch.utils.data import Dataset
from pathlib import Path
import pathlib
import torchaudio
from torchaudio.functional import resample

from torch.utils.data import DataLoader
import hydra
from hydra import initialize, compose
from hydra.utils import to_absolute_path

from demucs.states import get_quantizer
from demucs.demucs import Demucs
from demucs.hdemucs import HDemucs

import pytorch_lightning as pl


@hydra.main(config_path="conf", config_name="infer_config")
def main(args):    
    
    class InferDataset(Dataset):
        def __init__(self,
                     audio_folder_path,
                     audio_ext,
                     sampling_rate):

            audiofolder = Path(audio_folder_path) #path of folder
            self.audio_path_list = list(audiofolder.glob(f'*.{audio_ext}'))  #path of audio
            self.sample_rate = sampling_rate
            
            self.audio_name =[]
            for i in self.audio_path_list:
                path = pathlib.PurePath(i)
                self.audio_name.append(path.name)
                
        def __len__(self):
            return len(self.audio_path_list)

        def __getitem__(self, idx):       
            try:
                waveform, rate = torchaudio.load(self.audio_path_list[idx])
                #return (torch.Tensor, int)

                if rate!= self.sample_rate:
                    waveform = resample(waveform, rate,  self.sample_rate)  
                    #resample(waveform: torch.Tensor, orig_freq: int, new_freq: int)
                    #return waveform tensor at the new frequency of dimension 
            except:
                waveform = torch.tensor([[]])
                rate = 0
                print(f"{self.audio_path_list[idx].name} is corrupted") 
            audio_name = self.audio_name[idx] 
            return waveform, audio_name

    
    if args.checkpoint==None:
        raise ValueError("Please enter the path for your model checkpoint")
               
    if args.infer_audio_folder_path ==None:
        raise ValueError("Please enter the path for your inference audio folder") 
        
    inference_set = InferDataset(to_absolute_path(args.infer_audio_folder_path), args.infer_audio_ext, args.infer_samplerate)    
    inference_loader = DataLoader(inference_set, args.dataloader.inference.num_workers) 
    
    
    if args.model == 'Demucs':
        model = Demucs.load_from_checkpoint(to_absolute_path(args.checkpoint))
        #call with pretrained model
                
    elif args.model == 'HDemucs':
        model = HDemucs.load_from_checkpoint(to_absolute_path(args.checkpoint))
        
    else:
        print('Invalid model, please choose Demucs or HDemucs')    
    
    quantizer = get_quantizer(model, args.quant, model.optimizers)
    model.quantizer = quantizer #can use as self.quantizer in class Demucs
    
    trainer = pl.Trainer(**args.trainer)

    trainer.predict(model, dataloaders=inference_loader)
    
if __name__ == "__main__":
    main()      
    