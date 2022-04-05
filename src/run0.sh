CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.2 --val_mot20 True --experiments mot20_test_val_0.2_0.2 --chance test5 --score 0.2
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.2 --val_mot20 True --experiments mot20_avg --chance test5 
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.2 --val_mot20 True --experiments mot20_test_val_0.2_0.4 --chance test4 --score 0.4
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.2 --val_mot20 True --experiments mot20_test_val_0.2_0.5 --chance test4 --score 0.5
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.1 --val_mot20 True --experiments mot20_test_val_0.1_0.1 --chance test4 --score 0.1
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.1 --val_mot20 True --experiments mot20_test_val_0.1_0.2 --chance test4 --score 0.2
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.1 --val_mot20 True --experiments mot20_test_val_0.1_0.3 --chance test4 --score 0.3
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.1 --val_mot20 True --experiments mot20_test_val_0.1_0.4 --chance test4 --score 0.4
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.1 --val_mot20 True --experiments mot20_test_val_0.1_0.5 --chance test4 --score 0.5

CUDA_VISIBLE_DEVICES=1 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.2 --val_mot20 True --experiments mot20_test_1_avg_0.5 --chance test5 --Threshold 0.5

CUDA_VISIBLE_DEVICES=1 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.2 --val_mot20 True --experiments mot20_test_1_avg_0.5 --chance test5 --Threshold 0.5