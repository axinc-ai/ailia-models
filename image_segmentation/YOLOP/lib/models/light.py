import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized

CSPDarknet_s = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]]
]

# MCnet = [
# [ -1, Focus, [3, 32, 3]],
# [ -1, Conv, [32, 64, 3, 2]],
# [ -1, BottleneckCSP, [64, 64, 1]],
# [ -1, Conv, [64, 128, 3, 2]],
# [ -1, BottleneckCSP, [128, 128, 3]],
# [ -1, Conv, [128, 256, 3, 2]],
# [ -1, BottleneckCSP, [256, 256, 3]],
# [ -1, Conv, [256, 512, 3, 2]],
# [ -1, SPP, [512, 512, [5, 9, 13]]],
# [ -1, BottleneckCSP, [512, 512, 1, False]],
# [ -1, Conv,[512, 256, 1, 1]],
# [ -1, Upsample, [None, 2, 'nearest']],
# [ [-1, 6], Concat, [1]],
# [ -1, BottleneckCSP, [512, 256, 1, False]],
# [ -1, Conv, [256, 128, 1, 1]],
# [ -1, Upsample, [None, 2, 'nearest']],
# [ [-1,4], Concat, [1]],
# [ -1, BottleneckCSP, [256, 128, 1, False]],
# [ -1, Conv, [128, 128, 3, 2]],
# [ [-1, 14], Concat, [1]],
# [ -1, BottleneckCSP, [256, 256, 1, False]],
# [ -1, Conv, [256, 256, 3, 2]],
# [ [-1, 10], Concat, [1]],
# [ -1, BottleneckCSP, [512, 512, 1, False]],
# [ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],
# [ 17, Conv, [128, 64, 3, 1]],
# [ -1, Upsample, [None, 2, 'nearest']],
# [ [-1,2], Concat, [1]],
# [ -1, BottleneckCSP, [128, 64, 1, False]],
# [ -1, Conv, [64, 32, 3, 1]],
# [ -1, Upsample, [None, 2, 'nearest']],
# [ -1, Conv, [32, 16, 3, 1]],
# [ -1, BottleneckCSP, [16, 8, 1, False]],
# [ -1, Upsample, [None, 2, 'nearest']],
# [ -1, Conv, [8, 2, 3, 1]] #segmentation output
# ]

MCnet_SPP = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ -1, Conv,[512, 256, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1, 6], Concat, [1]],
[ -1, BottleneckCSP, [512, 256, 1, False]],
[ -1, Conv, [256, 128, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,4], Concat, [1]],
[ -1, BottleneckCSP, [256, 128, 1, False]],
[ -1, Conv, [128, 128, 3, 2]],
[ [-1, 14], Concat, [1]],
[ -1, BottleneckCSP, [256, 256, 1, False]],
[ -1, Conv, [256, 256, 3, 2]],
[ [-1, 10], Concat, [1]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
# [ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],
[ [17, 20, 23], Detect,  [13, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],
[ 17, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, SPP, [8, 2, [5, 9, 13]]] #segmentation output
]
# [2,6,3,9,5,13], [7,19,11,26,17,39], [28,64,44,103,61,183]
MCnet_fast = [
[ -1, Focus, [3, 32, 3]],#0
[ -1, Conv, [32, 64, 3, 2]],#1
[ -1, BottleneckCSP, [64, 128, 1, True, True]],#2
[ -1, BottleneckCSP, [128, 256, 1, True, True]],#4
[ -1, BottleneckCSP, [256, 512, 1, True, True]],#6
[ -1, SPP, [512, 512, [5, 9, 13]]],#8
[ -1, BottleneckCSP, [512, 512, 1, False]],#9
[ -1, Conv,[512, 256, 1, 1]],#10
[ -1, Upsample, [None, 2, 'nearest']],#11
[ [-1, 6], Concat, [1]],#12
[ -1, BottleneckCSP, [512, 256, 1, False]],#13
[ -1, Conv, [256, 128, 1, 1]],#14
[ -1, Upsample, [None, 2, 'nearest']],#15
[ [-1,4], Concat, [1]],#16
[ -1, BottleneckCSP, [256, 128, 1, False, True]],#17
[ [-1, 14], Concat, [1]],#19
[ -1, BottleneckCSP, [256, 256, 1, False, True]],#20
[ [-1, 10], Concat, [1]],#22
[ -1, BottleneckCSP, [512, 512, 1, False]],#23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 64, 3, 1]],#25
[ -1, Upsample, [None, 2, 'nearest']],#26
[ [-1,2], Concat, [1]],#27
[ -1, BottleneckCSP, [128, 32, 1, False]],#28
# [ -1, Conv, [64, 32, 1, 1]],#29
[ -1, Upsample, [None, 2, 'nearest']],#30
# [ -1, Conv, [32, 16, 1, 1]],#31
[ -1, BottleneckCSP, [32, 8, 1, False]],#32
[ -1, Upsample, [None, 2, 'nearest']],#33
[ -1, Conv, [8, 2, 1, 1]], #Driving area segmentation output#34

[ 16, Conv, [256, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 32, 1, False]],
# [ -1, Conv, [64, 32, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
# [ -1, Conv, [32, 16, 1, 1]],
[ 31, BottleneckCSP, [32, 8, 1, False]],#35
[ -1, Upsample, [None, 2, 'nearest']],#36
[ -1, Conv, [8, 2, 1, 1]], #Lane line segmentation output #37
]

MCnet_light = [
[ -1, Focus, [3, 32, 3]],#0
[ -1, Conv, [32, 64, 3, 2]],#1
[ -1, BottleneckCSP, [64, 64, 1]],#2
[ -1, Conv, [64, 128, 3, 2]],#3
[ -1, BottleneckCSP, [128, 128, 3]],#4
[ -1, Conv, [128, 256, 3, 2]],#5
[ -1, BottleneckCSP, [256, 256, 3]],#6
[ -1, Conv, [256, 512, 3, 2]],#7
[ -1, SPP, [512, 512, [5, 9, 13]]],#8
[ -1, BottleneckCSP, [512, 512, 1, False]],#9
[ -1, Conv,[512, 256, 1, 1]],#10
[ -1, Upsample, [None, 2, 'nearest']],#11
[ [-1, 6], Concat, [1]],#12
[ -1, BottleneckCSP, [512, 256, 1, False]],#13
[ -1, Conv, [256, 128, 1, 1]],#14
[ -1, Upsample, [None, 2, 'nearest']],#15
[ [-1,4], Concat, [1]],#16
[ -1, BottleneckCSP, [256, 128, 1, False]],#17
[ -1, Conv, [128, 128, 3, 2]],#18
[ [-1, 14], Concat, [1]],#19
[ -1, BottleneckCSP, [256, 256, 1, False]],#20
[ -1, Conv, [256, 256, 3, 2]],#21
[ [-1, 10], Concat, [1]],#22
[ -1, BottleneckCSP, [512, 512, 1, False]],#23
[ [17, 20, 23], Detect,  [1, [[4,12,6,18,10,27], [15,38,24,59,39,78], [51,125,73,168,97,292]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 128, 3, 1]],#25
[ -1, Upsample, [None, 2, 'nearest']],#26
# [ [-1,2], Concat, [1]],#27
[ -1, BottleneckCSP, [128, 64, 1, False]],#27
[ -1, Conv, [64, 32, 3, 1]],#28
[ -1, Upsample, [None, 2, 'nearest']],#29
[ -1, Conv, [32, 16, 3, 1]],#30
[ -1, BottleneckCSP, [16, 8, 1, False]],#31
[ -1, Upsample, [None, 2, 'nearest']],#32
[ -1, Conv, [8, 3, 3, 1]], #Driving area segmentation output#33

# [ 16, Conv, [128, 64, 3, 1]],
# [ -1, Upsample, [None, 2, 'nearest']],
# [ [-1,2], Concat, [1]],
# [ -1, BottleneckCSP, [128, 64, 1, False]],
# [ -1, Conv, [64, 32, 3, 1]],
# [ -1, Upsample, [None, 2, 'nearest']],
# [ -1, Conv, [32, 16, 3, 1]],
[ 30, BottleneckCSP, [16, 8, 1, False]],#34
[ -1, Upsample, [None, 2, 'nearest']],#35
[ -1, Conv, [8, 2, 3, 1]], #Lane line segmentation output #36
]


# The lane line and the driving area segment branches share information with each other
MCnet_share = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 64, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ [-1,2], Concat, [1]],  #27
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck

[ 16, Conv, [256, 64, 3, 1]],   #33
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ [-1,2], Concat, [1]], #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39   
[ -1, BottleneckCSP, [16, 8, 1, False]],    #40 lane line segment neck

[ [31,39], Concat, [1]],    #41
[ -1, Conv, [32, 8, 3, 1]],     #42    Share_Block


[ [32,42], Concat, [1]],     #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [16, 2, 3, 1]], #45 Driving area segmentation output


[ [40,42], Concat, [1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [16, 2, 3, 1]] #48Lane line segmentation output
]

# The lane line and the driving area segment branches without share information with each other
MCnet_no_share = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [13, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 64, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ [-1,2], Concat, [1]],  #27
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #33
[ -1, Conv, [8, 3, 3, 1]], #34 Driving area segmentation output

[ 16, Conv, [256, 64, 3, 1]],   #35
[ -1, Upsample, [None, 2, 'nearest']],  #36
[ [-1,2], Concat, [1]], #37
[ -1, BottleneckCSP, [128, 64, 1, False]],  #38
[ -1, Conv, [64, 32, 3, 1]],    #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, Conv, [32, 16, 3, 1]],    #41
[ -1, BottleneckCSP, [16, 8, 1, False]],    #42 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [8, 2, 3, 1]] #44 Lane line segmentation output
]



class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 13
        self.detector_index = -1
        self.Da_out_idx = 45 if len(block_cfg)==49 else 34
        # self.Da_out_idx = 37

        # Build model
        # print(block_cfg)
        for i, (from_, block, args) in enumerate(block_cfg):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                detects, _, _= self.forward(torch.zeros(1, 3, s, s))
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        #times = []
        for i, block in enumerate(self.model):
            #t0 = time_synchronized()
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            if isinstance(block, Detect):   # save detector result
                out.append(x)
            if i == self.Da_out_idx:     #save driving area segment result
                m=nn.Sigmoid()
                out.append(m(x))
            cache.append(x if block.index in self.save else None)
            """t1 = time_synchronized()
            print(str(i) + " : " + str(t1-t0))
            times.append(t1-t0)
        print(sum(times[:25]))
        print(sum(times[25:33]))
        print(sum(times[33:41]))
        print(sum(times[41:43]))
        print(sum(times[43:46]))
        print(sum(times[46:]))"""
        m=nn.Sigmoid()
        out.append(m(x))
        return out
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

class CSPDarknet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(CSPDarknet, self).__init__()
        layers, save= [], []
        # self.nc = 13    #output category num
        self.nc = 1
        self.detector_index = -1

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                detects, _ = self.forward(torch.zeros(1, 3, s, s))
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            start = time.time()
            x = block(x)
            end = time.time()
            print(start-end)
            """y = None if isinstance(x, list) else x.shape"""
            if isinstance(block, Detect):   # save detector result
                out.append(x)
            cache.append(x if block.index in self.save else None)
        m=nn.Sigmoid()
        out.append(m(x))
        # out.append(x)
        # print(out[0][0].shape, out[0][1].shape, out[0][2].shape)
        return out
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net(cfg, **kwargs): 
    # m_block_cfg = MCnet_share if cfg.MODEL.STRU_WITHSHARE else MCnet_no_share
    m_block_cfg = MCnet_no_share
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)

    detects, dring_area_seg, lane_line_seg = model(input_)
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(dring_area_seg.view(-1).shape)
    _,predict=torch.max(dring_area_seg, 1)
    print(predict.shape)
    print(lane_line_seg.shape)

    _,lane_line_pred=torch.max(lane_line_seg, 1)
    _,lane_line_gt=torch.max(gt_, 1)
    metric.reset()
    metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
    acc = metric.pixelAccuracy()
    meanAcc = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    IoU = metric.IntersectionOverUnion()
    print(IoU)
    print(mIoU)