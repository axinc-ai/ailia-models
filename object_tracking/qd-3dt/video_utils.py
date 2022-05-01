import json
from collections import defaultdict
import itertools

import numpy as np

from logging import getLogger  # noqa

logger = getLogger(__name__)


class VID:
    def __init__(self, dataset):
        self.dataset = dataset
        self.createIndex()
        self.parseInstances()

    def createIndex(self):
        logger.info('creating index...')

        anns, cats, imgs, videos = {}, {}, {}, {}
        videoToImgs, imgToAnns, catToImgs = \
            defaultdict(list), defaultdict(list), defaultdict(list)

        if 'videos' in self.dataset:
            for video in self.dataset['videos']:
                videos[video['id']] = video

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                videoToImgs[img['video_id']].append(img)
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        logger.info('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

        self.videos = videos
        self.videoToImgs = videoToImgs

    def parseInstances(self):
        instances = defaultdict(list)
        videoToInstanceIds = defaultdict(list)

        video_ids = self.getVidIds()
        for video_id in video_ids:
            tracklets = self.getInstanceFromVideoId(video_id)
            instances.update(tracklets)
            videoToInstanceIds[video_id] = list(tracklets.keys())

        self.instances = instances
        self.videoToInstanceIds = videoToInstanceIds

    def getInstanceFromVideoId(self, videoId):
        tracklets = defaultdict(list)
        img_ids = self.getImgIdsFromVidId(videoId)

        # tracklets = {instance_id: {video_id: 1, img_index: [], ann_ids: []}}
        for index, img_id in enumerate(img_ids):
            ann_ids = self.getAnnIds(img_id)
            anns = self.loadAnns(ann_ids)
            for ann in anns:
                instance_id = ann['instance_id']
                if instance_id not in tracklets.keys():
                    tracklets[instance_id] = defaultdict(list)
                    tracklets[instance_id]['video_id'] = videoId

                tracklets[instance_id]['img_indexes'].append(index)
                tracklets[instance_id]['ann_ids'].append(ann['id'])

        return tracklets

    def getAnnIds(
            self,
            imgIds=[], vidIds=[],
            catIds=[], areaRng=[],
            iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds \
            if isinstance(imgIds, tuple) or isinstance(imgIds, list) \
            else [imgIds]
        vidIds = vidIds \
            if isinstance(vidIds, tuple) or isinstance(vidIds, list) \
            else [vidIds]
        catIds = catIds \
            if isinstance(catIds, tuple) or isinstance(catIds, list) \
            else [catIds]

        if len(imgIds) == len(vidIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            imgIds = imgIds if (len(imgIds) > 0) or (len(vidIds) == 0) else \
                [i['id'] for vidId in vidIds for i in self.videoToImgs[vidId]]
            if not len(imgIds) == 0:
                lists = [
                    self.imgToAnns[imgId] for imgId in imgIds
                    if imgId in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [
                ann for ann in anns if ann['category_id'] in catIds
            ]
            anns = anns if len(areaRng) == 0 else [
                ann for ann in anns
                if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]
            ]

        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]

        return ids

    def getVidIds(self, videoIds=[]):
        videoIds = videoIds \
            if isinstance(videoIds, tuple) or isinstance(videoIds, list) \
            else [videoIds]

        if len(videoIds) == 0:
            ids = self.videos.keys()
        else:
            ids = set(videoIds)

        return list(ids)

    def getImgIdsFromVidId(self, videoId):
        img_infos = self.videoToImgs[videoId]
        ids = list(np.zeros([len(img_infos)], dtype=np.int))
        for img_info in img_infos:
            ids[img_info['index']] = img_info['id']
        return ids

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if isinstance(ids, tuple) or isinstance(ids, list):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if isinstance(ids, tuple) or isinstance(ids, list):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]


def load_annotations(ann_file):
    img_infos = []

    with open(ann_file, 'r') as f:
        dataset = json.load(f)

    vid = VID(dataset)
    vid_ids = vid.getVidIds()
    for vid_id in vid_ids:
        img_ids = vid.getImgIdsFromVidId(vid_id)
        for img_id in img_ids:
            info = vid.loadImgs([img_id])[0]
            info['filename'] = info['file_name']
            info['type'] = 'VID'
            info['first_frame'] = True if info['index'] == 0 else False
            img_infos.append(info)

    return img_infos
