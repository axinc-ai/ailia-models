import argparse
import os
import sys

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    self.parser.add_argument('--exp_id', default = 'default')
    self.parser.add_argument('--gpus', default='0', help='-1 for CPU')
    self.parser.add_argument('--num_workers', type=int, default=4)
    self.parser.add_argument('--test', action = 'store_true', help = 'test')
    self.parser.add_argument('--debug', type = int, default = 0)
    self.parser.add_argument('--demo', default = '../images/', help = 'path/to/image')

    self.parser.add_argument('--task', default='human2d')
    self.parser.add_argument('--ratio_3d', type=float, default=0)
    self.parser.add_argument('--weight_3d', type=float, default=0)
    self.parser.add_argument('--weight_var', type=float, default=0)
    self.parser.add_argument('--full_test', action='store_true')


    self.parser.add_argument('--hide_data_time', action = 'store_true')
    self.parser.add_argument('--metric', default = 'acc')
    self.parser.add_argument('--resume', action = 'store_true')
    self.parser.add_argument('--load_model', default = '')
    self.parser.add_argument('--weight_decay', type=float, default=0.0)
    self.parser.add_argument('--scale', type=float, default=-1)
    self.parser.add_argument('--rotate', type=float, default=-1)
    self.parser.add_argument('--flip', type = float, default=0.5)
    self.parser.add_argument('--dataset', default = 'mpii', 
                             help = 'mpii | coco')
    self.parser.add_argument('--all_pts', action = 'store_true',
                             help = 'heatmap for all persons in stack 1')
    self.parser.add_argument('--multi_person', action = 'store_true', 
                             help = 'heatmap for all persons in final stack')
    self.parser.add_argument('--fit_short_side', action = 'store_true', 
                             help = 'fit to long or short bbox side when'
                                    'the input resolution is rectangle')
    self.parser.add_argument('--lr', type=float, default=0.001)
    self.parser.add_argument('--lr_step', type=str, default='90,120')
    self.parser.add_argument('--num_epochs', type=int, default=140)
    self.parser.add_argument('--val_intervals', type=int, default=5)
    self.parser.add_argument('--batch_size', type=int, default=32)
    self.parser.add_argument('--arch', default = 'msra_50', 
                             help = 'hg | msra_xxx')
    self.parser.add_argument('--disable_cudnn', action = 'store_true')
    self.parser.add_argument('--save_all_models', action = 'store_true')
    self.parser.add_argument('--print_iter', type = int, default = -1, 
                             help = 'for run in cloud server')

    self.parser.add_argument('--input_h', type = int, default = -1)
    self.parser.add_argument('--input_w', type = int, default = -1)
    self.parser.add_argument('--output_h', type = int, default = -1)
    self.parser.add_argument('--output_w', type = int, default = -1)

  def parse(self, args = ''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)
    
    opt.eps = 1e-6
    opt.momentum = 0.0
    opt.alpha = 0.99
    opt.epsilon = 1e-8
    opt.hm_gauss = 2
    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp')

    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    if opt.debug > 0:
      opt.num_workers = 1

    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    if opt.test:
      opt.exp_id = opt.exp_id + 'TEST'
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)

    if 'hg' in opt.arch or 'posenet' in opt.arch:
      opt.num_stacks = 2
    else:
      opt.num_stacks = 1
    
    if opt.input_h == -1 and opt.input_w == -1 and \
      opt.output_h == -1 and opt.output_w == -1:
      if opt.dataset == 'coco':
        opt.input_h, opt.input_w = 256, 192
        opt.output_h, opt.output_w = 64, 48
      else:
        opt.input_h, opt.input_w = 256, 256
        opt.output_h, opt.output_w = 64, 64
    else:
      assert opt.input_h // opt.output_h == opt.input_w // opt.output_w
    
    if opt.scale == -1:
      opt.scale = 0.3 if opt.dataset == 'coco' else 0.25
    if opt.rotate == -1:
      opt.rotate = 40 if opt.dataset == 'coco' else 30

    opt.num_output = 17 if opt.dataset == 'coco' else 16
    opt.num_output_depth = opt.num_output if opt.task == 'human3d' else 0
    opt.heads = {'hm': opt.num_output}
    if opt.num_output_depth > 0:
      opt.heads['depth'] = opt.num_output_depth
    print('heads', opt.heads)


    if opt.resume:
      opt.load_model = '{}/model_last.pth'.format(opt.save_dir)

    return opt
