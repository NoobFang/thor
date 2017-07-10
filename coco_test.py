from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pylab
pylab.rcParams['figure.figsize'] = (10.0,8.0)

dataDir = '/home/fhy/datasets/coco'
dataType = 'train2014'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

thor_sp = ['appliance', 'electronic', 'food', 'furniture', 'indoor', 'kitchen']
cats = coco.loadCats(coco.getCatIds())
thor_cats = [cat['name'] for cat in cats if cat['supercategory'] in thor_sp]
catIds = coco.getCatIds(catNms=thor_cats)
imgIds = coco.getImgIds(catIds=catIds)
imgs = coco.loadImgs(imgIds)
#img = imgs[np.random.randint(0,len(imgs))]
#filename = '{}/train2014/{}'.format(dataDir, img['file_name'])
#I = cv2.imread(filename)
annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds)
anns = coco.loadAnns(annIds)
'''
plt.figure(); plt.axis('off')
plt.imshow(I)
coco.showAnns(anns)
plt.show()
'''