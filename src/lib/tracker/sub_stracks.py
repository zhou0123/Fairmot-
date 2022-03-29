import itertools
import os
import os.path as osp
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.decode import mot_decode,mot_decode_
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat,_tranpose_and_gather_feat_expand
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process

from tracker import matching

from .basetrack import BaseTrack, TrackState

#最新的拓展
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps as bbox_ious
from matplotlib import pyplot as plt
import copy
from tracking_utils import kalman_filter
import lap

class STrack_f5(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self,opt,tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self.opt = opt
        self._tlwh = np.asarray(tlwh[0,:], dtype=np.float)

        self._tlwhs = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        
        self.index_feat = None
        self.add_feat = None # 位置 匹配 未匹配

        self.smooth_feat = None
        self.feat_pool =None
        self.new_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
    def add_new_features(self,f_nums = None):
        
        """
        f_nums : 未选上的feature的索引
        """

        not_selected = self.new_feat[f_nums,:]
        ln = np.linalg.norm(not_selected,axis=1).reshape(-1,1)
        not_selected /= ln

        if self.feat_pool is None:
            self.feat_pool = not_selected
            loc = np.arange(len(self.feat_pool))
            nums = np.zeros(len(self.feat_pool))
            self.add_feat = np.vstack((loc,nums,nums)).T.astype(int)
        else:
            # 先做相似度的计算，然后决定vstack还是update
            cost_matrix = np.maximum(0.0, cdist(self.feat_pool, not_selected, 'cosine'))
           
            matches, u_pool, u_not_selected = matching.linear_assignment(cost_matrix, thresh=0.4)
            if len(matches)!=0:
                pool , nselected = matches[:,0],matches[:,1]

                self.feat_pool[pool,:] = self.alpha * self.feat_pool[pool,:] + (1 - self.alpha) * not_selected[nselected,:]
                self.add_feat[pool,1]+=1
            loc1 = np.arange(len(u_not_selected))+len(self.feat_pool)
            nums1 = np.zeros(len(loc1))
            sub_add = np.vstack((loc1,nums1,nums1)).T
            self.feat_pool = np.vstack((self.feat_pool,not_selected[u_not_selected,:]))
            self.add_feat[u_pool,1]+=1
            self.add_feat = np.vstack((self.add_feat,sub_add))

            #增加到smooth_feat

            add = np.where(self.add_feat[:,1]>self.opt.add_TF)[0]
            loc2 = np.arange(len(add))+len(self.smooth_feat)
            nums2= np.zeros(len(loc2))
            sub_add2 = np.vstack((loc2,nums2)).T
            self.index_feat=np.vstack((self.index_feat,sub_add2))
            self.smooth_feat = np.vstack((self.smooth_feat,self.feat_pool[add,:]))
            
            
            add = set(add)
            all_ = set(np.arange(len(self.add_feat)))
            leave = np.array(list(add^all_)).astype(int)
            self.add_feat = self.add_feat[leave,:]

            #删除feat_pool
            delete = np.where(self.add_feat[:,2]>self.opt.del_TF)[0]
            delete = set(delete)
            all_ = set(np.arange(len(self.add_feat)))
            leave = np.array(list(delete^all_)).astype(int)
            self.add_feat = self.add_feat[leave,:].astype(int)
            if len(self.add_feat) == 0:
                self.feat_pool =None
                return
            # print("aqa"*10)
            # print(self.feat_pool.shape)
            # print(self.add_feat.shape)
            # print(self.add_feat)
            self.feat_pool = self.feat_pool[self.add_feat[:,0],:]
            self.add_feat[:,0] = np.arange(len(self.add_feat))

    def monitor(self,nums=None):

        if nums is None:
            return
        
        loc = set(self.index_feat[:,0])
        nums = set (nums)
        no_appear  = np.array(list(loc ^ nums)).astype(int)
        self.index_feat[no_appear,1]+=1
        #删除的环节

        remain = np.where(self.index_feat[:,1]<5)[0]
        self.index_feat = self.index_feat[remain,:]
        self.index_feat[:,0] = np.arange(len(self.index_feat))
        self.smooth_feat = self.smooth_feat[remain,:]
       

    def update_features(self, feat,nums=None):
        """
        feat [x,128]
        """
        ln = np.linalg.norm(feat,axis=1).reshape(-1,1)
        feat /= ln
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
            loc = np.arange(len(self.smooth_feat))
            num = np.zeros(len(self.smooth_feat))
            self.index_feat = np.vstack((loc,num)).T.astype(int)
        else:
            self.smooth_feat[nums,:] = self.alpha * self.smooth_feat[nums,:] + (1 - self.alpha) * feat
        #self.features.append(feat)
        if nums is None:
            ln = np.linalg.norm(self.smooth_feat,axis=1).reshape(-1,1)
            self.smooth_feat /= ln            
        else:
            ln = np.linalg.norm(self.smooth_feat[nums,:],axis=1).reshape(-1,1)
            self.smooth_feat[nums,:] /= ln
        self.monitor(nums)
            

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, record,new_id=False,update_feature= False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        if update_feature:
            record = np.array(record)
            self.update_features(new_track.curr_feat[record[:,1],:],record[:,0])
        # self.smooth_feat = new_track.curr_feat
        # loc = np.arange(len(self.smooth_feat))
        # num = np.zeros(len(self.smooth_feat))
        # self.index_feat = np.vstack((loc,num)).T.astype(int)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id,record=None,update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.new_feat=new_track.curr_feat
            record = np.array(record)
            self.update_features(new_track.curr_feat[record[:,1],:],record[:,0])
            selected = set(record[:,1])
            all_ = set(np.arange(len(new_track.curr_feat)))
            f_nums = np.array(list(selected^all_)).astype(int)
            self.add_new_features(f_nums)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    def tlbr_to_tlwh_f5(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[:,2:] -= ret[:,:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)




