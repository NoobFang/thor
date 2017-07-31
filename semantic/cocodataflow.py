import numpy as np
import tensorflow as tf
from tensorpack import *
from pycocotools.coco import COCO
import cv2
import os

class CocoDataFlow(DataFlow):
  def __init__(self, dataDir, spFilter, shuffle=True):
    assert os.path.isdir(dataDir), dataDir
    self.dir = dataDir
    annFile = '{}/annotations/instances_train2014.json'.format(self.dir)
    self.coco = COCO(annFile)
    self.allCats = self.coco.loadCats(self.coco.getCatIds())
    self.catNms = [cat['name'] for cat in self.allCats
                 if cat['supercategory'] in spFilter]
    self.catIds = self.coco.getCatIds(catNms=self.catNms)
    self.imgIds = self.coco.getImgIds(catIds=self.catIds)
    self.imgs = self.coco.loadImgs(self.imgIds)
    self.num_cat = len(self.catNms)
    self.cids = {}  # the map from category id to mask channel id
    i = 0
    for c in self.catIds:
      self.cids[c] = i
      i += 1
    assert len(self.cids) == self.num_cat
    self.shuffle = shuffle


  def get_data(self):
    idxs = np.arange(len(self.imgs))
    if self.shuffle:
      np.random.shuffle(idxs)
    for idx in idxs:
      img = self.imgs[idx]
      annIds = self.coco.getAnnIds(imgIds=[img['id']], catIds=self.catIds)
      anns = self.coco.loadAnns(annIds)
      fname = '{}/train2014/{}'.format(self.dir, img['file_name'])
      I = cv2.imread(fname, cv2.IMREAD_COLOR)
      assert I is not None
      h, w = I.shape[0], I.shape[1]
      label = self.preprocess_ann(anns, h, w)

      yield cv2.resize(I, (300,300)), cv2.resize(label, (300,300))

  def size(self):
    return len(self.imgs)

  def reset_state(self):
    pass

  def preprocess_ann(self, anns, h, w):
    mask = np.zeros(shape=(h, w, self.num_cat))
    for ann in anns:
      cid = self.cids[ann['category_id']]
      m = self.coco.annToMask(ann)
      mask[:,:,cid] = m
    return mask

if __name__ == '__main__':
  import time
  spFilter = ['appliance', 'electronic', 'food', 'furniture', 'indoor', 'kitchen']
  dataset = CocoData(dataDir='/home/fhy/datasets/coco', spFilter=spFilter)
  dataset = BatchData(dataset, 128)

  flow = dataset.get_data()
  start = time.time()
  for i in range(10):
    img, label = next(flow)
  end = time.time()
  print('{} secs per datapoint'.format((end-start)/10/128))