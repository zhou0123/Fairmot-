CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot16 True --experiments mot16_test_val --chance orign 
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot16 True --experiments mot16_test_val_0.95 --chance test2 --score 0.95
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot16 True --experiments mot16_test_val_0.9 --chance test2 --score 0.9
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot16 True --experiments mot16_test_val_0.85 --chance test2 --score 0.85
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot16 True --experiments mot16_test_val_0.8 --chance test2 --score 0.8


