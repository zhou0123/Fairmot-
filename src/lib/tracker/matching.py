import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracking_utils import kalman_filter
import time
def nms_gather(opt,dets,id_featrues):
    """
    dets [ x , 5]
    id_features [ x, 128]
    """

    Threshold=opt.Threshold

    x1,y1,x2,y2,scores= dets[:,0],dets[:,1],dets[:,2],dets[:,3],dets[:,4]
    areas=(y2-y1+1)*(x2-x1+1)
    orders=(-1*scores).argsort()

    keep=[]
    keep_nums=[]
    keep_dets = []
    keep_features = []
    while orders.shape[0]>0:
        
        i=orders[0]
        
        keep.append(i)
        
        xx1=np.maximum(x1[i],x1[orders[1:]])
        yy1=np.maximum(y1[i],y1[orders[1:]])
        xx2=np.minimum(x2[i],x2[orders[1:]])
        yy2=np.minimum(y2[i],y2[orders[1:]])
        
        ws=np.maximum(xx2-xx1+1,0)
        hs=np.maximum(yy2-yy1+1,0)
        
        inter=ws*hs
        iou=inter/(areas[i]+areas[orders[1:]]-inter)
        
        index=np.where(iou<Threshold)[0]

        gather = np.where(iou>=Threshold)[0]

        gather_index = orders[gather+1]
        keep_dets.append(np.vstack((dets[i,:],dets[gather_index,:])))
        array = np.vstack((id_featrues[i,:],id_featrues[gather_index,:]))
        keep_features.append(array)
        keep_nums.append(len(array))
        orders=orders[index+1]
    
    return keep,keep_nums,keep_dets,keep_features
def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix
def iou_distance_(atracks, btracks,cost_matrix,keep_nums,strack_nums,lambda_ = 0.98):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)

    index1 = np.where(_ious < 0.3)
    index2 = np.where(_ious>=0.3)

    _ious[index1] = np.inf
    _ious[index2] = 0.0

    start = 0
    for row, track in enumerate(atracks):

        gating_distance = _ious[row]
        gating_distance = np.repeat(gating_distance,keep_nums)
        cost_matrix[start : start + strack_nums[row]] = lambda_ * cost_matrix[start : start + strack_nums[row]] + (1 - lambda_) * gating_distance
        start = start + strack_nums[row]
    return cost_matrix

def embedding_distance_(tracks, detections, ind=None,metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
   
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    det_features=det_features.transpose(1,2,0)
    _,_,nums = det_features.shape
    det_features = det_features[ind,:,np.arange(nums)]
    print("det_features.shape",det_features.shape)
    
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    
    track_features=track_features.transpose(1,2,0)
    _,_,nums_ = track_features.shape
    track_features = track_features[ind,:,np.arange(nums_)]
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix
def iou_next(tracks,dets_all,tracks_features,H,W):
    track_det = np.asarray([track.tlbr() for track in tracks],dtype=np.float)

    ious_ = ious(track_det,track_det)

    ious_[np.arange(len(ious_)),np.arange(len(ious_))] = 0

    ind = ious_.argmax(axis=1)

    ind2_ = ious_[np.arange(len(ious_)),ind]>0.8

    ind2 = np.arange(len(ious_))[ind2_]  #第几个track
    ind = ind[ind2_]

    for ind_,ind2_ in zip(ind,ind2):
        track = tracks[ind_]
        near_track = tracks[ind2_]

        track.near_id = near_track.track_id
        num = track.ind
        ind_around = np.array([num-2-2*W,num-1-2*W,num-2*W,num+1-2*W,num+2-2*W,\
            num-2-W,num-1-W,num-W,num+1-W,num+2-W,\
            num-2,num-1,num,num+1,num+2,\
            num-2+W,num-1+W,num+W,num+1+W,num+2+W,\
            num-2+2*W,num-1+2*W,num+2*W,num+1+2*W,num+2+2*W
            ]).reshape(-1,)
        ind_around = np.clip(ind_around,0,len(dets_all)-1).astype(int)
        track.around_feats = tracks_features[ind_around]

        num = near_track.ind
        ind_around = np.array([num-2-2*W,num-1-2*W,num-2*W,num+1-2*W,num+2-2*W,\
            num-2-W,num-1-W,num-W,num+1-W,num+2-W,\
            num-2,num-1,num,num+1,num+2,\
            num-2+W,num-1+W,num+W,num+1+W,num+2+W,\
            num-2+2*W,num-1+2*W,num+2*W,num+1+2*W,num+2+2*W
            ]).reshape(-1,)
        ind_around = np.clip(ind_around,0,len(dets_all)-1).astype(int)
        near_track.around_feats = tracks_features[ind_around]







def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    id_27 = []
    id_243 = []

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    # id_27 = np.asarray([track.smooth_feat for track in tracks if track.track_id == 27], dtype=np.float)
    # id_243 = np.asarray([track.smooth_feat for track in tracks if track.track_id == 243], dtype=np.float)
    # if len(id_27) !=0 and len(id_243) !=0:
    #     print("aqa"*10)
    #     print("id_27",id_27)
    #     print("id_243",id_243)
    #     print("now")
    #     #print(np.maximum(0.0, cdist(np.split(id_27,16), np.split(id_243,16), 'cosine')))
    #     print(np.maximum(0.0, cdist(id_27.reshape(1,-1), id_243.reshape(1,-1), metric)))
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix

def embedding_distance_f5(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    strack_nums = [len(track.smooth_feat) for track in tracks]
    keep_nums = [len(track.curr_feat) for track in detections]
    if cost_matrix.size == 0:
        return cost_matrix,strack_nums,keep_nums
    det_features = np.vstack([track.curr_feat for track in detections]).astype(np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.vstack([track.smooth_feat for track in tracks]).astype(np.float)
    
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix,strack_nums,keep_nums


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

def fuse_motion_f5(kf, cost_matrix, tracks, detections,keep_nums,strack_nums,only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    start=0
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        gating_distance = np.repeat(gating_distance,keep_nums)
        index_ = np.where(gating_distance > gating_threshold)[0]
        cost_matrix[start : start + strack_nums[row], index_] = np.inf
        #gating_distance = np.repeat(gating_distance,keep_nums)
        cost_matrix[start : start + strack_nums[row]] = lambda_ * cost_matrix[start : start + strack_nums[row]] + (1 - lambda_) * gating_distance

        start = start + strack_nums[row]
    return cost_matrix

