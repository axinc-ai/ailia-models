# bdd_labels = {
# 'unlabeled':0, 'dynamic': 1, 'ego vehicle': 2, 'ground': 3, 
# 'static': 4, 'parking': 5, 'rail track': 6, 'road': 7, 
# 'sidewalk': 8, 'bridge': 9, 'building': 10, 'fence': 11, 
# 'garage': 12, 'guard rail': 13, 'tunnel': 14, 'wall': 15,
# 'banner': 16, 'billboard': 17, 'lane divider': 18,'parking sign': 19, 
# 'pole': 20, 'polegroup': 21, 'street light': 22, 'traffic cone': 23,
# 'traffic device': 24, 'traffic light': 25, 'traffic sign': 26, 'traffic sign frame': 27,
# 'terrain': 28, 'vegetation': 29, 'sky': 30, 'person': 31,
# 'rider': 32, 'bicycle': 33, 'bus': 34, 'car': 35, 
# 'caravan': 36, 'motorcycle': 37, 'trailer': 38, 'train': 39,
# 'truck': 40       
# }
id_dict = {'person': 0, 'rider': 1, 'car': 2, 'bus': 3, 'truck': 4, 
'bike': 5, 'motor': 6, 'tl_green': 7, 'tl_red': 8, 
'tl_yellow': 9, 'tl_none': 10, 'traffic sign': 11, 'train': 12}
id_dict_single = {'car': 0, 'bus': 1, 'truck': 2,'train': 3}
# id_dict = {'car': 0, 'bus': 1, 'truck': 2}

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
