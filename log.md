1 尝试feature-id尽可能的不同
2 feature-id中新加入的或者某个feature-id，与之对应的相似度最高的小于某个阈值，在备选的9宫格中选择最低的

命令行：
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot15 True
CUDA_VISIBLE_DEVICES=0 python track.py mot --load_model /home/zhouchengyu/paper1/others/fairmot_dla34.pth --conf_thres 0.4 --val_mot16 True --experiments mot16_test_val
test : mot_15
原本：
IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP  FN IDs  FM   MOTA  MOTP IDt IDa IDm
Venice-2 53.1% 40.3% 77.8% 91.6% 47.4% 26 23  3  0 7263 602  57  93 -10.9% 0.203  10  12   2

3 尝试调参，dcos的权重