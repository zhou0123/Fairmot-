CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.3 --val_mot20 True --experiments mot20_test_val_0.5 --chance test4 --score 0.5
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.3 --val_mot20 True --experiments mot20_test_val_0.45 --chance test4 --score 0.45
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.3 --val_mot20 True --experiments mot20_test_val_0.4 --chance test4 --score 0.4
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.3 --val_mot20 True --experiments mot20_test_val_0.35 --chance test4 --score 0.35
CUDA_VISIBLE_DEVICES=2 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.3 --val_mot20 True --experiments mot20_test_val_0.3 --chance test4 --score 0.3

