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
from tracker import sub_stracks
from .sub_stracks import STrack_f5
import pandas as pd
import lap




class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

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

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
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
            self.update_features(new_track.curr_feat)

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
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class Detections_(object):
    def __init__(self,opt,heatmap,features,dets,frame_rate=30):
        """
        heatmap [B,C,H,W]
        feature [B,C(ids),H,W]
        """
        self.opt=opt
        self.heatmap=heatmap.cpu().numpy()
        self.features=features.cpu().numpy()
        self.dets=dets
        self.tracks_features=None
        self.nums=0
        self.W=0
        self.H=0
    def get_around_features_boxes(self,index):
        all_=[]
        scores=[]
        boxes=[]
        w,h=index%self.W,index//self.W
        for i in range(self.opt.size):
            for j in range(self.opt.size):
                h_index=torch.clip(h-self.opt.size//2+i, min=0, max=self.H-1)
                w_index=torch.clip(h-self.opt.size//2+j, min=0, max=self.W-1)
                all_.append(self.features[:,:,h_index,w_index])
                scores.append(self.heatmap[:,:,h_index,w_index])
                index_box=torch.clip((h-self.opt.size//2+i)*self.W+(w-self.opt.size//2+j), min=0, max=self.H*self.W-1)
                boxes.append(self.dets[index_box,:])
        return all_,scores,boxes
    def distribute_id(self,index,feature,det):
        B,C,self.H,self.W=self.heatmap.shape
        if self.nums==0:
            self.tracks_features=feature.reshape(1,-1)
            w,h=index%self.W,index//self.W
            s=self.heatmap[:,:,h,w]
            self.nums+=1
            return feature,np.concatenate((det,s[0]))
        all_,scores,boxes=self.get_around_features_boxes(index)
        outs=[]
        for feature_,score_,det_ in zip(all_,scores,boxes):
            cost=np.maximum(0.0, cdist(feature_, self.tracks_features, 'cosine'))
            det1=np.ascontiguousarray(det.reshape(1,-1), dtype=np.float)
            det_=np.ascontiguousarray(np.maximum(0.0,det_).reshape(1,-1), dtype=np.float)
            if np.max(cost)<self.opt.diff_degree or bbox_ious(det1,det_)<0.9:
                outs.append(-1)
                continue
            cost = np.sum(cost*score_)
            outs.append(cost)
        if sum(outs)==-9:
            self.tracks_features=np.vstack((self.tracks_features,feature))
            w,h=index%self.W,index//self.W
            s=self.heatmap[:,:,h,w]
            return feature,np.concatenate((det,s[0]))
        index_max=outs.index(max(outs))
        self.tracks_features=np.vstack((self.tracks_features,all_[index_max]))
        
        return all_[index_max][0],np.concatenate((boxes[index_max],np.array(scores[index_max][0])))
class check_cos(object):
    def __init__(self,opt,heat,boxes,tracks_features):
        self.opt=opt
        self.heat=heat.cpu().numpy()
        self.boxes=boxes
        self.tracks_features=tracks_features.cpu().numpy()
        self.features=None
        self.num=0
        self.H,self.W=heat.size(2),heat.size(3)
    def check_and_revise(self,feature,ind,det):
        s=self.heat[:,:,ind//self.W,ind%self.W]
        if self.num==0:
            self.features=feature.reshape(1,-1)
            self.num+=1
            return feature,np.concatenate((det,s[0]))
        cost=np.min(cdist(feature.reshape(1,-1),self.features, 'cosine'))
        w,h=ind%self.W,ind//self.W
        if cost<self.opt.diff_weight:
            costs=[]
            all_=[]
            boxes=[]
            scores=[]
            for i in range(self.opt.size):
                for j in range(self.opt.size):

                    h_index=torch.clip(h-self.opt.size//2+i, min=0, max=self.H-1)
                    w_index=torch.clip(h-self.opt.size//2+j, min=0, max=self.W-1)
                    f=self.tracks_features[:,:,h_index,w_index]
                    all_.append(f)
                    scores.append(self.heat[:,:,h_index,w_index])
                    index_box=torch.clip((h-self.opt.size//2+i)*self.W+(w-self.opt.size//2+j), min=0, max=self.H*self.W-1)
                    boxes.append(self.boxes[index_box,:])
                    

                    if self.heat[:,:,h_index,w_index]<self.heat[:,:,h,w]-self.opt.diff_degree:
                        costs.append(-1)
                    else:
                        cost=np.min(cdist(f,self.features,"cosine"))
                        costs.append(cost)
            
            index_=costs.index(max(costs))
            self.features=np.vstack((self.features,all_[index_]))
            return all_[index_],np.concatenate((boxes[index_],np.array(scores[index_][0])))
        
        self.features=np.vstack((self.features,feature))

        return feature,np.concatenate((det,s[0]))
class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results
    def merge_outputs_(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks
    def update_f5(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)
            reg = output['reg'] if self.opt.reg_offset else None

            dets_all= mot_decode_(hm,wh,reg=reg,ltrb=self.opt.ltrb)
            tracks_features=F.normalize(output['id'], dim=1).cpu().numpy().squeeze(0).transpose(1,2,0).reshape(-1,128)
        dets_all = self.post_process(dets_all, meta)
        dets_all = self.merge_outputs_([dets_all])[1]
        
        #dets_all.shape,tracks_features.shape (41344, 5) (41344, 128)

        #remain_inds = dets_all[:, 4] > self.opt.conf_thres #0.2 
        remain_inds = dets_all[:, 4] > 0.3
        #sum(remain_inds) 378
        dets_all = dets_all[remain_inds]
        tracks_features = tracks_features[remain_inds]

        keep,keep_nums,keep_dets,keep_features = matching.nms_gather(self.opt,dets_all,tracks_features)
        #print("keep_nums",len(keep_nums))
        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''
        if len(keep) > 0:
            '''Detections'''
            detections = [STrack_f5(self.opt,STrack_f5.tlbr_to_tlwh_f5(tlbrs[:,:4]), tlbrs[0,4], f, 30) for
                          (tlbrs, f) in zip(keep_dets, keep_features)]
        else:
            detections = []
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists,strack_nums,keep_nums= matching.embedding_distance_f5(strack_pool, detections)

        dists = matching.fuse_motion_f5(self.kalman_filter, dists, strack_pool, detections,keep_nums,strack_nums) # 选择最高分box的作为惩罚相加
        #非avg
        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)
        # matches, u_track, u_detection,record = conform(strack_nums,keep_nums,matches)
        #非avg
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)
        matches, u_track, u_detection,record=conform_avg(strack_nums,keep_nums,matches,dists)
        for i in range(len(record)):
            itracked, idet = matches[i]
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id,record[i])
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, record[i],new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        #print("len(u_detection)",len(u_detection))
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id,update_feature=False)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id,new_id=False,update_feature=False)
                refind_stracks.append(track)
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        #print("len(u_detection)",len(u_detection))
        '''Deal with unconfirmed tracks'''
        detections = [detections[i] for i in u_detection]
        dists1,strack_nums,keep_nums= matching.embedding_distance_f5(unconfirmed, detections)
        dists = matching.iou_distance_(unconfirmed, detections,dists1,keep_nums,strack_nums)
        #dists = matching.fuse_motion_f5(self.kalman_filter, dists, strack_pool, detections,keep_nums,strack_nums)
        
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.4)
        #matches, u_unconfirmed, u_detection,record = conform(strack_nums,keep_nums,matches)
        
        matches, u_unconfirmed, u_detection,record=conform_avg(strack_nums,keep_nums,matches,dists)
        
        for i in range(len(record)):
            itracked, idet = matches[i]
            unconfirmed[itracked].update(detections[idet], self.frame_id,record[i])
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks
    def update_f4(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]
        _,_,h_,w_=hm.shape
        
        scores=hm[0,0,inds[0]//w_,inds[0]%w_]
        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]
        scores=scores[remain_inds]
       
        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)

        lis_det=[]
        # print(scores)
        # print("first:",len(strack_pool))
        # print("first:",len(detections))
        for i in range(len(detections)):
            track_features = np.asarray([track.smooth_feat for track in strack_pool], dtype=np.float)
            if len(track_features) == 0:
                break
            score = scores[i]
            if score <self.opt.score:
                break
            id_feature_ = id_feature[i].reshape(1,-1)
            det = detections[i]
            dists = np.maximum(0.0, cdist(track_features, id_feature_, "cosine"))
            
            
            gating_dim=2
            gating_threshold = kalman_filter.chi2inv95[gating_dim]
            measurements = np.asarray([det.to_xyah()])

            for row, track in enumerate(strack_pool):
                gating_distance = self.kalman_filter.gating_distance(
                    track.mean, track.covariance, measurements, False, metric='maha')
                dists[row, gating_distance > gating_threshold] = np.inf
                lambda_=0.98
                dists[row] = lambda_ * dists[row] + (1 - lambda_) * gating_distance
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)
            
            for itracked, idet in matches:
                track = strack_pool[itracked]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            if len(u_detection) == 0:
                lis_det.append(i)
                strack_pool.pop(itracked)
                #print("a"*100)
                continue
            ''' Step 3: Second association, with IOU'''
            # detections = [detections[i] for i in u_detection]
            # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            atlbrs = [track.tlbr for track in strack_pool]
            btlbrs=[det.tlbr]
            dists = 1-matching.ious(atlbrs, btlbrs)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

            for itracked, idet in matches:
                track = strack_pool[itracked]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            if len(u_detection) == 0:
                lis_det.append(i)
                strack_pool.pop(itracked)
                continue
            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            #detections = [detections[i] for i in u_detection]
            atlbrs = [track.tlbr for track in unconfirmed]
            dists = 1-matching.ious(atlbrs, btlbrs)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                unconfirmed[itracked].update(det, self.frame_id)
                activated_starcks.append(unconfirmed[itracked])
            if len(u_detection) == 0:
                lis_det.append(i)
                unconfirmed.pop(itracked)
        detections = np.delete(detections,lis_det).tolist()
        # print("second:",len(strack_pool))
        # print("second:",len(detections))
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks
    def update_f3(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_features = []
            hms=[]
            for i in range(3):
                for j in range(3):
                    id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0).cpu().numpy()
                    hm_=_tranpose_and_gather_feat_expand(hm, inds, bias=(i - 1, j - 1)).squeeze(0).cpu().numpy()
                    id_features.append(id_feature_exp)
                    hms.append(hm_)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]
        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]
            hms[i]=hms[i][remain_inds]

        
        id_features=np.array(id_features).transpose(1,0,2)
        hms=np.array(hms).transpose(1,0,2).squeeze()

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_features)]
        else:
            detections = []
        indexs=hms.argsort(axis=1)
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        for x in range(9):

            index_x=indexs[:,x]
            
            if np.sum(hms[np.arange(len(index_x)),index_x]>=0.7)!=len(index_x) and x != 0:
                strack_pool=strack_pool_last
                break

            dists = matching.embedding_distance(strack_pool, detections,index_x)
            dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                num=index_x[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id,num)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            strack_pool_last=copy.deepcopy(strack_pool)
            strack_pool = [strack_pool[i] for i in u_track]
            detections = [detections[i] for i in u_detection]
            

        ''' Step 3: Second association, with IOU'''
        #detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks
    def update_f2(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_features = []
            for i in range(3):
                for j in range(3):
                    id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0).cpu().numpy()
                    id_features.append(id_feature_exp)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]
        
        #plt.imsave("/home/zhouchengyu/hms/hm_{}.png".format(self.frame_id),hm.cpu().numpy()[0,0,:,:])
        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]
        #new:所有boxes
        #feature_id 的初始化
        if self.frame_id==1:
            FEATURES=[]
            DETS=[]
            dets_all= mot_decode_(hm,wh,reg=reg,ltrb=self.opt.ltrb)
            dets_all = self.post_process(dets_all, meta)
            dets_all = self.merge_outputs_([dets_all])[1][:,:4]
            
            tracks_features=F.normalize(output['id'], dim=1)
            check = check_cos(self.opt,hm,dets_all,tracks_features)
            for tlbr,f,index in zip(dets[:,:4],id_feature,inds[0][remain_inds]):
                
                fea,de=check.check_and_revise(f,index,tlbr)
                fea=fea.reshape(-1,)
                FEATURES.append(fea)
                DETS.append(de)
            
            id_feature=np.array(FEATURES)
            dets=np.array(DETS)
        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)

        #这里加入优先级最高的算法:
        # for track in strack_pool:
        #     print(track.smooth_feat.shape)
        strack_pool_features=np.array([track.smooth_feat for track in strack_pool], dtype=np.float)
        record_strack={}
        record_loc={}
        if len(strack_pool_features)!=0:
            for i in range(len(id_feature)):

                fea_s=np.array([f[i] for f in id_features])
                ds=cdist(fea_s,strack_pool_features,"cosine")
                
                min_=np.min(ds)
                if min_<self.opt.min_level:
                    h,w=np.where(min_==ds)
                    h=h[0]
                    w=w[0]
                    if w in record_strack:
                        record_strack[w]= -1
                        record_loc[w]=h
                    else:
                        record_strack[w]=i
                        record_loc[w]=h
    
            keys=[]
            locs=[]
            for key in record_strack.keys():
                #key 是 old

                if record_strack[key] != -1:

                    num = record_loc[key]
                    loc = record_strack[key]
                    if num == 4:

                        track = strack_pool[key]
                        det = detections[loc]
                        keys.append(key)
                        locs.append(loc)
                        if track.state == TrackState.Tracked:
                            track.update(det, self.frame_id)
                            activated_starcks.append(track)
                        else:
                            track.re_activate(det, self.frame_id, new_id=False)
                            refind_stracks.append(track)
                    else:
                        #如果不是最中心的位置，则不会更新
                        track = strack_pool[key]
                        det = detections[loc]
                        keys.append(key)
                        locs.append(loc)
                        if track.state == TrackState.Tracked:
                            track.update(det, self.frame_id,update_feature=False)
                            activated_starcks.append(track)
                        else:
                            track.re_activate(det, self.frame_id, new_id=False)
                            refind_stracks.append(track)

            strack_pool=np.delete(np.array(strack_pool),keys).tolist()
            detections=np.delete(np.array(detections),locs).tolist()


        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks
    def update_f1(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)
            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            #全部boxes
            dets_all= mot_decode_(hm,wh,reg=reg,ltrb=self.opt.ltrb)

            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()
        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]
        #dets=dets[0][:,:5].cpu().numpy()
        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]
        
        #add new 
        dets_all = self.post_process(dets_all, meta)
        dets_all = self.merge_outputs_([dets_all])[1][:,:4]
        """
        这里采用我的方法
        """
        
        features=F.normalize(output['id'],dim=1)
        init=Detections_(self.opt,hm,features,dets_all)
        FEATURES=[]
        DETS=[]
        for tlbr,f,index in zip(dets[:,:4],id_feature,inds[0][remain_inds]):
            fea,de=init.distribute_id(index,f,tlbr)
            FEATURES.append(fea)
            DETS.append(de)
        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''
        # if len(dets) > 0:
        
        #     '''Detections'''
        #     detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
        #                   (tlbrs, f) in zip(dets[:, :5], id_feature)]
       
        if len(DETS) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(DETS, FEATURES)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def conform(strack_nums,keep_nums,matches):
    results = []
    u_track= []
    u_detection = []
    record = []
    start_ = 0
    if len(strack_nums) == 0 or len(matches)==0:
        u_detection = [ i for i in range(len(keep_nums))]
        return np.array(results), u_track ,u_detection,record
    for _ in range(len(strack_nums)):
        end_ = strack_nums[_] + start_
        track_where = np.where((matches[:,0]>=start_)& (matches[:,0]<end_))[0]
        dets_target = pd.Series(matches[track_where,:][:,1]) # 判断区间
        dets_target_=matches[track_where,:]
        is_betweens=[]
        se=[]
        start = 0
        start__ = start_
        start_ = end_ 
        for i in range(len(keep_nums)):
            end = start+keep_nums[i]
            se.append([start,end])#左闭右开
            is_between = dets_target.between(start, end-1)
            is_betweens.append(np.sum(np.array(is_between)))
            start = end
        out = is_betweens.index(max(is_betweens))
        if max(is_betweens)>len(track_where)/2:
            is_between = dets_target.between(se[out][0],se[out][1]-1)
            dets_target = dets_target_[is_between,:]-[start__,se[out][0]]
            record.append(dets_target)
            results.append([_,out])
        
    results = np.array(results)
    if len(results) == 0:
        return results, u_track ,u_detection,record
    for i in range(len(strack_nums)):
        if i in results[:,0]:
            continue
        u_track.append(i)
    for i in range(len(keep_nums)):
        if i in results[:,1]:
            continue
        u_detection.append(i)
    return results, u_track ,u_detection,record
def conform_avg(strack_nums,keep_nums,matches,dists):
    start = 0
    all_ = []
    record_all = []
    record = []
    u_track= []
    u_detection = []
    if len(strack_nums) == 0 or len(matches)==0:
        u_detection = [ i for i in range(len(keep_nums))]
        return np.array([]), u_track ,u_detection,record
    for i in range(len(strack_nums)):
        end = strack_nums[i]+start
        track_where = np.where((matches[:,0]>=start)& (matches[:,0]<end))[0]
        if len(track_where)==0:
            all_.extend([np.inf]*(len(keep_nums)))
            start = end
            record_all.append([np.array([]).reshape(0,2)]*len(keep_nums))
            continue
        start__ = start
        start = end
        start_ = 0 
        record_sub = []
        for j in range(len(keep_nums)):
            end_ = keep_nums[j]+start_
            k1 = matches[track_where,:]
            keep_where = np.where((k1[:,1]>=start_)&(k1[:,1]<end_))[0]
            k1 =k1[keep_where,:]
            record_sub.append(np.array(k1-[start__,start_]))
            if len(k1) == 0:
                num = np.inf
            else:
                #num = np.sum(dists[k1[:,0],k1[:,1]])/len(k1)
                num = min(dists[k1[:,0],k1[:,1]])
            all_.append(num) 
            start_ = end_
        record_all.append(record_sub)
    
    all_ = np.array(all_).reshape(len(strack_nums),len(keep_nums))
    matches = []
    cost_ ,s,k = lap.lapjv(all_,extend_cost=True, cost_limit=0.4)
    for ix, mx in enumerate(s):
        if mx >= 0:
            matches.append([ix, mx])
    u_track = np.where(s < 0)[0]
    u_detection = np.where(k < 0)[0]
    matches = np.asarray(matches)
    x = np.where(s!=-1)[0]
    s_ = s[x]
    record_all = np.array(record_all,dtype=object)
    record = record_all[np.arange(len(record_all))[x],s_].tolist()
    return matches, u_track ,u_detection,record





            



    

    





        
            







    
