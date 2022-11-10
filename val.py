import os
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable

from hydra.utils import to_absolute_path as abs_path
from utils.utils import pt_eval, getpoints


def eval_net(net, loader, device, n_val, val_pair, epoch, cfg):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    total_mse = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, (imgs1, imgs2, imgs3, imgs4, gtnum, gts, gt1234) in enumerate(loader):

            output_dir = abs_path(f'val_pts/pts_batch{cfg.eval.batch_size}/epoch{epoch+1}')
            os.makedirs(output_dir) if os.path.isdir(output_dir) is False else None

            for j in range(len(imgs1)):
                imgs1[j] = Variable(imgs1[j].to(device=device))
                imgs2[j] = Variable(imgs2[j].to(device=device))
                imgs3[j] = Variable(imgs3[j].to(device=device))
                imgs4[j] = Variable(imgs4[j].to(device=device))
            
            imgs1 = torch.cat(imgs1, dim=0)
            imgs2 = torch.cat(imgs2, dim=0)
            imgs3 = torch.cat(imgs3, dim=0)
            imgs4 = torch.cat(imgs4, dim=0)

            with torch.no_grad():       
                hm1 = net(imgs1)[0].cpu().numpy()
                hm2 = net(imgs2)[0].cpu().numpy()
                hm3 = net(imgs3)[0].cpu().numpy()
                hm4 = net(imgs4)[0].cpu().numpy()

            mags_pred1 = np.transpose(hm1, axes=[0, 2, 3, 1])
            mags_pred2 = np.transpose(hm2, axes=[0, 2, 3, 1])
            mags_pred3 = np.transpose(hm3, axes=[0, 2, 3, 1])
            mags_pred4 = np.transpose(hm4, axes=[0, 2, 3, 1])

            if imgs1.shape[0] == 1:
                data12 = np.hstack((mags_pred1, mags_pred2))
                data34 = np.hstack((mags_pred3, mags_pred4))
                out_data = np.vstack((data12, data34))
            else:
                data12 = np.concatenate([mags_pred1, mags_pred2], axis=2)
                data34 = np.concatenate([mags_pred3, mags_pred4], axis=2)
                out_data = np.concatenate([data12, data34], axis=1)

            pre_pos = []
            for ii in range(imgs1.shape[0]):
                pre_pos.append(np.array(getpoints(out_data[ii, :, :, 0])))

            pbar.update(imgs1.shape[0])

            save_path = []      
            for index in range(imgs1.shape[0]):
                save_path.append(abs_path(output_dir + val_pair[i*imgs1.shape[0]+index][0][-13:-4] + '_loc.txt'))

            gtm = gts[0].numpy()
            for batch in range(imgs1.shape[0]):
                sc = []
                for ii in pre_pos[batch]:
                    sc.append(out_data[batch][int(ii[1]), int(ii[0])])

                if len(pre_pos[batch])==0:
                    gx = []
                    gtnum = 0
                    for k in range(len(gts[0])):
                        if gtm[batch][k][0] > 0:
                            gx.append(gtm[batch][k][0])
                    gtnum += len(gx)
                else:
                    gtnum, _, _, pts = pt_eval(gts[0][batch].numpy()[np.newaxis], pre_pos[batch][:, 0], pre_pos[batch][:, 1], np.array(sc)[:, 0])

                if len(pre_pos[batch])>0:
                    np.savetxt(save_path[batch], pts)
        
    return total_mse
