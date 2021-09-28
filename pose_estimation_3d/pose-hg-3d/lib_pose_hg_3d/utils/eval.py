import numpy as np

def get_preds(hm, return_conf=False):
  assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
  h = hm.shape[2]
  w = hm.shape[3]
  hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
  idx = np.argmax(hm, axis = 2)
  
  preds = np.zeros((hm.shape[0], hm.shape[1], 2))
  for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
      preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
  if return_conf:
    conf = np.amax(hm, axis = 2).reshape(hm.shape[0], hm.shape[1], 1)
    return preds, conf
  else:
    return preds

def calc_dists(preds, gt, normalize):
  dists = np.zeros((preds.shape[1], preds.shape[0]))
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
        dists[j][i] = \
          ((gt[i][j] - preds[i][j]) ** 2).sum() ** 0.5 / normalize[i]
      else:
        dists[j][i] = -1
  return dists

def dist_accuracy(dist, thr=0.5):
  dist = dist[dist != -1]
  if len(dist) > 0:
    return 1.0 * (dist < thr).sum() / len(dist)
  else:
    return -1

def accuracy(output, target, acc_idxs):
  preds = get_preds(output)
  gt = get_preds(target)
  dists = calc_dists(preds, gt, np.ones(target.shape[0]) * target.shape[2] / 10)
  acc = np.zeros(len(acc_idxs))
  avg_acc = 0
  bad_idx_count = 0
  
  for i in range(len(acc_idxs)):
    acc[i] = dist_accuracy(dists[acc_idxs[i]])
    if acc[i] >= 0:
      avg_acc = avg_acc + acc[i]
    else:
      bad_idx_count = bad_idx_count + 1
  
  if bad_idx_count == len(acc_idxs):
    return 0
  else:
    return avg_acc / (len(acc_idxs) - bad_idx_count)

def get_preds_3d(heatmap, depthmap):
  output_res = max(heatmap.shape[2], heatmap.shape[3])
  preds = get_preds(heatmap).astype(np.int32)
  preds_3d = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.float32)
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      idx = min(j, depthmap.shape[1] - 1)
      pt = preds[i, j]
      preds_3d[i, j, 2] = depthmap[i, idx, pt[1], pt[0]]
      preds_3d[i, j, :2] = 1.0 * preds[i, j] / output_res
    preds_3d[i] = preds_3d[i] - preds_3d[i, 6:7]
  return preds_3d


def mpjpe(heatmap, depthmap, gt_3d, convert_func):
  preds_3d = get_preds_3d(heatmap, depthmap)
  cnt, pjpe = 0, 0
  for i in range(preds_3d.shape[0]):
    if gt_3d[i].sum() ** 2 > 0:
      cnt += 1
      pred_3d_h36m = convert_func(preds_3d[i])
      err = (((gt_3d[i] - pred_3d_h36m) ** 2).sum(axis=1) ** 0.5).mean()
      pjpe += err
  if cnt > 0:
    pjpe /= cnt
  return pjpe, cnt  

  