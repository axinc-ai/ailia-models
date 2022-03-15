import random
import shutil
import os

def split(path, mask_path, lane_path):
    os.mkdir(path + 'train')
    os.mkdir(path + 'val')
    os.mkdir(mask_path + 'train')
    os.mkdir(mask_path + 'val')
    os.mkdir(lane_path + 'train')
    os.mkdir(lane_path + 'val')
    val_index = random.sample(range(660), 200)
    for i in range(660):
        if i in val_index:
            shutil.move(path+'{}.png'.format(i), path + 'val')
            shutil.move(mask_path+'{}.png'.format(i), mask_path + 'val')
            shutil.move(lane_path+'{}.png'.format(i), lane_path + 'val')
        else:
            shutil.move(path+'{}.png'.format(i), path + 'train')
            shutil.move(mask_path+'{}.png'.format(i), mask_path + 'train')
            shutil.move(lane_path+'{}.png'.format(i), lane_path + 'train')


if __name__ == '__main__':
    path = "/home/wqm/bdd/data_hust/"
    mask_path = "/home/wqm/bdd/hust_area/"
    lane_path = "/home/wqm/bdd/hust_lane/"
    split(path, mask_path, lane_path)


