# IDA-SiamNet
This is a PyTorch implementation of the paper IDA-SiamNet: Interactive- and Dynamic-Aware Siamese Network for Building Change Detection

Train
```
python tools/train.py configs/idasiamnet/idasiamnet_ex_r18_512x512_40k_levircd.py --work-dir ./idasiam_r18_levir

python tools/train.py configs/idasiamnet/idasiamnet_ex_mit-b1_512x512_40k_levircd.py --work-dir ./idasiam_mit_levir

python tools/train.py configs/idasiamnet/idasiamnet_ex_r18_512x512_80k_s2looking.py --work-dir ./idasiamnet_r18_s2looking

python tools/train.py configs/idasiamnet/idasiamnet_ex_mit-b1_512x512_80k_s2looking.py --work-dir ./idasiamnet_mit_s2looking

python tools/train.py configs/idasiamnet/idasiamnet_ex_r18_256x256_100e_whucd.py --work-dir ./idasiamnet_r18_whucd

python tools/train.py configs/idasiamnet/idasiamnet_ex_mit-b1_256x256_100e_whucd.py --work-dir ./idasiamnet_mit_whucd
```

Infer
```
python tools/test.py configs/idasiamnet/idasiamnet_ex_r18_512x512_40k_levircd.py idasiam_r18_levir/best_mIoU_iter_40000.pth

python tools/test.py configs/idasiamnet/idasiamnet_ex_mit-b1_512x512_40k_levircd.py idasiam_mit_levir/best_mIoU_iter_36000.pth

python tools/test.py configs/idasiamnet/idasiamnet_ex_r18_512x512_80k_s2looking.py idasiamnet_r18_s2looking/best_mIoU_iter_80000.pth

python tools/test.py configs/idasiamnet/idasiamnet_ex_mit-b1_512x512_80k_s2looking.py idasiamnet_mit_s2looking/best_mIoU_iter_80000.pth

python tools/test.py configs/idasiamnet/idasiamnet_ex_r18_256x256_100e_whucd.py idasiamnet_r18_whucd/best_mIoU_epoch_100.pth

python tools/test.py configs/idasiamnet/idasiamnet_ex_mit-b1_256x256_100e_whucd.py idasiamnet_mit_whucd/best_mIoU_epoch_100.pth
```
