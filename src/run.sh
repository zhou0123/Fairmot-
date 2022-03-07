CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot16 True --experiments mot16_test_val_0.1 --chance test2 --min_level 0.1
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot16 True --experiments mot16_test_val_0.2 --chance test2 --min_level 0.2
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot16 True --experiments mot16_test_val_0.3 --chance test2 --min_level 0.3


