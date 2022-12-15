# Multi-Frame Attention with Feature-Level Warping for Drone Crowd Tracking

## Overview of our method
![Illustration](./image/overview.png)
This repository provides the offical release of the code package for my paper Multi-Frame Attention with Feature-Level Warping for Drone Crowd Tracking in WACV 2022 (url comming soon). 

</ul>
<table>
<thead>
<tr>
<th align="center"></th>
<th align="center">Tracking-mAP</th>
<th align="center">Localization-mAP</th>

</tr>
<tr>
<td align="center">Heamap (CVPR17)</td>
<td align="center">31.44</td>
<td align="center">29.26</td>
</tr> 

<tr>
<td align="center">Heamap + Ours</td>
<td align="center"><strong>33.25</strong></td>
<td align="center"><strong>32.19</strong></td>
</tr> 

<tr>
<td align="center"><strong>MPM</strong></td>
<td align="center"><strong>41.07</strong></td>
<td align="center"><strong>43.43</strong></td>
</tr>    

<tr>
<td align="center"><strong>MPM + Ours</strong></td>
<td align="center"><strong>41.91</strong></td>
<td align="center"><strong>42.08</strong></td>
</tr>    

</tbody></table>

## Requirements
```
$ pip install -r requirements.py
```

## Create Ground-Truth
If you use this code, run this command to create heatmap ground-truth first.
Training & Validation ground-truth are added in dataset directory.
```
$ python create_gts.py
```

## Training
### Example
```
$ python train.py
```
## Dataset
### DroneCrowd (Full Version)
This full version consists of 112 video clips with 33,600 high resolution frames (i.e., 1920x1080) captured in 70 different scenarios.  With intensive amount of effort, our dataset provides 20,800 people trajectories with 4.8 million head annotations and several video-level attributes in sequences.  

DroneCrowd [BaiduYun](https://pan.baidu.com/s/1hjXoVZJ16y9Tf7UXcJw3oQ)(code:ml1u)| [GoogleDrive](https://drive.google.com/drive/folders/1EUKLJ1WmrhWTNGt4wFLyHRfspJAt56WN?usp=sharing) 


## Citation

Please cite this paper if you want to use it in your work.
```
@inproceedings{dronecrowd_cvpr2021,
  author    = {Longyin Wen and
               Dawei Du and
               Pengfei Zhu and
               Qinghua Hu and
               Qilong Wang and
               Liefeng Bo and
               Siwei Lyu},
  title     = {Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark},
  booktitle = {CVPR},
  year      = {2021}
}
```
```
@article{zhu2021graph,
  title={Graph Regularized Flow Attention Network for Video Animal Counting from Drones},
  author={Zhu, Pengfei and Peng, Tao and Du, Dawei and Yu, Hongtao and Zhang, Libo and Hu, Qinghua},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```
