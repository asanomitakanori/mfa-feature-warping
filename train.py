import logging
import os
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from val import eval_net
from Model.model import MFAmodel as model

import hydra
from hydra.utils import to_absolute_path as abs_path

from utils.batchs_sampler import SequentialSampler
from utils.dataset_train import Heatmap_Dataset
import utils.dataset_val as Dataset
from utils.val_path import val_path
from utils.seed import *


def train_net(net,
              device,
              cfg,
              val_clip
              ):
              
    if cfg.eval.imgs is not None:
        train = Heatmap_Dataset(cfg.train, cfg.dataloader)
        val = Dataset.listDataset(val_clip, shuffle=False, transform=transforms.Compose([transforms.ToTensor()]), train=False)
        n_train = len(train)
        n_val = len(val)
    else:
        dataset = Heatmap_Dataset(cfg.train, cfg.dataloader)
        n_val = int(len(dataset) * cfg.eval.rate)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])

    set_seed(cfg.train.seed)
    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size
    sampler_train = SequentialSampler(train, cfg.train.batch_size, shuffle = True)
    train_loader = DataLoader(train, batch_size=cfg.train.video_number, worker_init_fn = worker_init_fn(cfg.train.seed), sampler = sampler_train, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=cfg.train.batch_size, num_workers=3, pin_memory=True, drop_last=True) 

    writer = SummaryWriter(log_dir=abs_path('./logs'), comment=f'LR_{cfg.train.lr}_BS_{batch_size}')
    global_step = 0

    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    criterion = nn.MSELoss()

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Sub Batch size:  {batch_size}
        Learning rate:   {cfg.train.lr}
        Training size:   {len(train)}
        Validation size: {len(val)}
        Checkpoints:     {cfg.output.save}
        Device:          {device.type}
        Optimizer        {optimizer.__class__.__name__}
        Criterion        {criterion.__class__.__name__}
    ''')

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['img']
                mag_gt = batch['mag']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mag_gt = mag_gt.to(device=device, dtype=torch.float32)

                V, B, C, H, W = imgs.shape
                Vm, Bm, Cm, Hm, Wm = mag_gt.shape     

                imgs = imgs.reshape(V*B, C, H, W)
                mag_gt = mag_gt.reshape(Vm*Bm, Cm, Hm, Wm)

                mags_pred, _ = net(imgs)
                loss = criterion(mags_pred, mag_gt) 
                epoch_loss += loss.item()

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % int(n_train // imgs.shape[0]) == 0:
                    val_loss = eval_net(net, val_loader, device, n_val, val_clip, epoch, cfg)

        if cfg.output.save:
            try:
                os.mkdir(abs_path(cfg.output.dir))
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       abs_path(os.path.join(cfg.output.dir, f'CP_epoch{epoch + 1}.pth')))
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()


@hydra.main(config_path='config', config_name='train')
def main(cfg):
    set_seed(cfg.train.seed)
    val_clip = val_path(cfg.eval.imgs)

    cfg.output.dir = cfg.output.dir + "_batch" + str(cfg.train.batch_size) 
    cfg.output.dir = f'model_param/' + cfg.output.dir 
    os.makedirs(abs_path(cfg.output.dir)) if os.path.isdir(abs_path(cfg.output.dir)) is False else None

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = model(n_channels=3, n_classes=1, num=8, sub_batch = cfg.train.batch_size, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if cfg.load:
        net.load_state_dict(
            torch.load(cfg.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.load}')

    net.to(device=device)
    
    try:
        train_net(net=net,
                  device=device,
                  cfg=cfg,
                  val_clip=val_clip)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), abs_path('INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
    main()
