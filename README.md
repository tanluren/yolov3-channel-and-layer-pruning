# yolov3-channel-and-layer-pruning
本项目以[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为基础实现，根据论文[Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)原理基于bn层Gmma系数进行通道剪枝，下面引用了几种不同的通道剪枝策略，并对原策略进行了改进，提高了剪枝率和精度；在这些工作基础上，又衍生出了层剪枝，本身通道剪枝已经大大减小了模型参数和计算量，降低了模型对资源的占用，而层剪枝可以进一步减小了计算量，并大大提高了模型推理速度；通过层剪枝和通道剪枝结合，可以压缩模型的深度和宽度，某种意义上实现了针对不同数据集的小模型搜索。<br>
<br>
项目的基本工作流程是，使用yolov3训练自己数据集，达到理想精度后进行稀疏训练，稀疏训练是重中之重，对需要剪枝的层对应的bn gamma系数进行大幅压缩，理想的压缩情况如下图，然后就可以对不重要的通道或者层进行剪枝，剪枝后可以对模型进行微调恢复精度，后续会写篇博客记录一些实验过程及调参经验，在此感谢[行云大佬](https://github.com/zbyuan)的讨论和合作！<br>
<br>
![稀疏](https://github.com/tanluren/yolov3-channel-and-layer-pruning/blob/master/data/img/1.jpg)

<br>

####  更新
1.增加了对**yolov3-spp**结构的支持，基础训练可以直接使用yolov3-spp.weights初始化权重，各个层剪枝及通道剪枝脚本的使用也和yolov3一致。<br>
2.增加了多尺度推理支持，train.py和各剪枝脚本都可以指定命令行参数, 如 --img_size 608 .<br>
3.2019/12/06更改了层剪枝的选层策略，由最大值排序改为均值排序。<br>
4.2019/12/08**重要**更新，增加了**知识蒸馏**策略。蒸馏是用高精度的大模型指导低精度的小模型，在结构相似的情况下效果尤为明显。而剪枝得到的小模型和原模型在结构上高度相似，非常符合蒸馏的应用条件。这里更新了一个参考Hinton大神Distilling the Knowledge in a Neural Network的蒸馏策略，原策略是针对分类模型的，但在这里也有不错的表现。调用只需要在微调的时候指定老师模型的cfg和权重即可：--t_cfg  --t_weights。最近会更新第二种针对yolo检测的知识蒸馏策略。<br>
5.2019/12/10交流的小伙伴比较多，回答不过来，可以加群734912150 <br>
6.2019/12/14增加了针对蒸馏的混合精度训练支持，项目中各项训练都可以使用[apex](https://github.com/NVIDIA/apex)加速,但需要先安装。使用混合精度可以加速训练，同时减轻显存占用，但训练效果可能会差一丢丢。代码默认开启了混合精度，如需关闭，可以把train.py中的mixed_precision改为False.<br>
7.2019/12/23更新了**知识蒸馏策略二**，并默认使用二。策略二参考了论文"Learning Efficient Object Detection Models with Knowledge Distillation"，相比策略一，对分类和回归分别作了处理，分类的蒸馏和策略一差不多，回归部分会分别计算学生和老师相对target的L2距离，如果学生更远，学生会再向target学习，而不是向老师学习。调用同样是指定老师的cfg和权重即可。需要强调的是，蒸馏在这里只是辅助微调，如果注重精度优先，剪枝时尽量剪不掉点的比例，这时蒸馏的作用也不大；如果注重速度，剪枝比例较大，导致模型精度下降较多，可以结合蒸馏提升精度。<br>
8.2019/12/27更新了两种**稀疏策略**，详看下面稀疏训练环节。<br>
9.2020/01/02修正各剪枝版本多分辨率推理test问题，主要是把命令行参数img_size传递给test函数。<br>
10.2020/01/04补了个[博客](https://blog.csdn.net/weixin_41397123/article/details/103828931)分享**无人机数据集visdrone**案例，演示如何压缩一个12M的无人机视角目标检测模型（标题党）。<br>
11.2020/04/10增加了**yolov3-tiny**的剪枝支持，稀疏照旧，剪通道用slim_prune.py，不可剪层。<br>
12.2020/4/24增加支持**yolov4**剪枝.<br>
13.2020/4/30在datasets.py 592行添加了支持负样本训练，默认注释掉.<br>
14.2020/7/8更新支持**yolov4-tiny**剪通道.<br>



#### 基础训练
环境配置查看requirements.txt，数据准备参考[这里](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)，预训练权重可以从darknet官网下载。<br>
用yolov3训练自己的数据集，修改cfg，配置好data，用yolov3.weights初始化权重。<br>
<br>
`python train.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/yolov3.weights --epochs 100 --batch-size 32`

#### 稀疏训练
scale参数默认0.001，根据数据集，mAP,BN分布调整，数据分布广类别多的，或者稀疏时掉点厉害的适当调小s;-sr用于开启稀疏训练；--prune 0适用于prune.py，--prune 1 适用于其他剪枝策略。稀疏训练就是精度和稀疏度的博弈过程，如何寻找好的策略让稀疏后的模型保持高精度同时实现高稀疏度是值得研究的问题，大的s一般稀疏较快但精度掉的快，小的s一般稀疏较慢但精度掉的慢；配合大学习率会稀疏加快，后期小学习率有助于精度回升。<br>
注意：训练保存的pt权重包含epoch信息，可通过`python -c "from models import *; convert('cfg/yolov3.cfg', 'weights/last.pt')"`转换为darknet weights去除掉epoch信息，使用darknet weights从epoch 0开始稀疏训练。<br>
<br>
`python train.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.weights --epochs 300 --batch-size 32 -sr --s 0.001 --prune 1`
* ##### 稀疏策略一：恒定s
这是一开始的策略，也是默认的策略。在整个稀疏过程中，始终以恒定的s给模型添加额外的梯度，因为力度比较均匀，往往压缩度较高。但稀疏过程是个博弈过程，我们不仅想要较高的压缩度，也想要在学习率下降后恢复足够的精度，不同的s最后稀疏结果也不同，想要找到合适的s往往需要较高的时间成本。<br>
<br>
`bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))`
* ##### 稀疏策略二：全局s衰减
关键代码是下面这句，在epochs的0.5阶段s衰减100倍。前提是0.5之前权重已经完成大幅压缩，这时对s衰减有助于精度快速回升，但是相应的bn会出现一定膨胀，降低压缩度，有利有弊，可以说是牺牲较大的压缩度换取较高的精度，同时减少寻找s的时间成本。当然这个0.5和100可以自己调整。注意也不能为了在前半部分加快压缩bn而大大提高s，过大的s会导致模型精度下降厉害，且s衰减后也无法恢复。如果想使用这个策略，可以在prune_utils.py中的BNOptimizer把下面这句取消注释。<br>
<br>
`# s = s if epoch <= opt.epochs * 0.5 else s * 0.01`
* ##### 稀疏策略三：局部s衰减
关键代码是下面两句，在epochs的0.5阶段开始对85%的通道保持原力度压缩，15%的通道进行s衰减100倍。这个85%是个先验知识，是由策略一稀疏后尝试剪通道几乎不掉点的最大比例，几乎不掉点指的是相对稀疏后精度；如果微调后还是不及baseline，或者说达不到精度要求，就可以使用策略三进行局部s衰减，从中间开始重新稀疏，这可以在牺牲较小压缩度情况下提高较大精度。如果想使用这个策略可以在train.py中把下面这两句取消注释，并根据自己策略一情况把0.85改为自己的比例，还有0.5和100也是可调的。策略二和三不建议一起用，除非你想做组合策略。<br>
<br>
`#if opt.sr and opt.prune==1 and epoch > opt.epochs * 0.5:`<br>
`#  idx2mask = get_mask2(model, prune_idx, 0.85)`

#### 通道剪枝策略一
策略源自[Lam1360/YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning)，这是一种保守的策略，因为yolov3中有五组共23处shortcut连接，对应的是add操作，通道剪枝后如何保证shortcut的两个输入维度一致，这是必须考虑的问题。而Lam1360/YOLOv3-model-pruning对shortcut直连的层不进行剪枝，避免了维度处理问题，但它同样实现了较高剪枝率，对模型参数的减小有很大帮助。虽然它剪枝率最低，但是它对剪枝各细节的处理非常优雅，后面的代码也较多参考了原始项目。在本项目中还更改了它的阈值规则，可以设置更高的剪枝阈值。<br>
<br>
`python prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --percent 0.85`

#### 通道剪枝策略二
策略源自[coldlarry/YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning)，这个策略对涉及shortcut的卷积层也进行了剪枝，剪枝采用每组shortcut中第一个卷积层的mask，一共使用五种mask实现了五组shortcut相关卷积层的剪枝，进一步提高了剪枝率。本项目中对涉及shortcut的剪枝后激活偏移值处理进行了完善，并修改了阈值规则，可以设置更高剪枝率，当然剪枝率的设置和剪枝后的精度变化跟稀疏训练有很大关系，这里再次强调稀疏训练的重要性。<br>
<br>
`python shortcut_prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --percent 0.6`

#### 通道剪枝策略三
策略参考自[PengyiZhang/SlimYOLOv3](https://github.com/PengyiZhang/SlimYOLOv3)，这个策略的通道剪枝率最高，先以全局阈值找出各卷积层的mask，然后对于每组shortcut，它将相连的各卷积层的剪枝mask取并集，用merge后的mask进行剪枝，这样对每一个相关层都做了考虑，同时它还对每一个层的保留通道做了限制，实验中它的剪枝效果最好。在本项目中还对激活偏移值添加了处理，降低剪枝时的精度损失。<br>
<br>
`python slim_prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --global_percent 0.8 --layer_keep 0.01`

#### 层剪枝
这个策略是在之前的通道剪枝策略基础上衍生出来的，针对每一个shortcut层前一个CBL进行评价，对各层的Gmma均值进行排序，取最小的进行层剪枝。为保证yolov3结构完整，这里每剪一个shortcut结构，会同时剪掉一个shortcut层和它前面的两个卷积层。是的，这里只考虑剪主干中的shortcut模块。但是yolov3中有23处shortcut，剪掉8个shortcut就是剪掉了24个层，剪掉16个shortcut就是剪掉了48个层，总共有69个层的剪层空间；实验中对简单的数据集剪掉了较多shortcut而精度降低很少。<br>
<br>
`python layer_prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --shortcuts 12`

#### 同时剪层和通道
前面的通道剪枝和层剪枝已经分别压缩了模型的宽度和深度，可以自由搭配使用，甚至迭代式剪枝，调配出针对自己数据集的一副良药。这里整合了一个同时剪层和通道的脚本，方便对比剪枝效果，有需要的可以使用这个脚本进行剪枝。<br>
<br>
`python layer_channel_prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --shortcuts 12 --global_percent 0.8 --layer_keep 0.1`

#### 微调finetune
剪枝的效果好不好首先还是要看稀疏情况，而不同的剪枝策略和阈值设置在剪枝后的效果表现也不一样，有时剪枝后模型精度甚至可能上升，而一般而言剪枝会损害模型精度，这时候需要对剪枝后的模型进行微调，让精度回升。训练代码中默认了前6个epoch进行warmup，这对微调有好处，有需要的可以自行调整超参学习率。<br>
<br>
`python train.py --cfg cfg/prune_0.85_my_cfg.cfg --data data/my_data.data --weights weights/prune_0.85_last.weights --epochs 100 --batch-size 32`

#### tensorboard实时查看训练过程
`tensorboard --logdir runs`<br>
<br>
![tensorboard](https://github.com/tanluren/yolov3-channel-and-layer-pruning/blob/master/data/img/2.jpg)
<br>
欢迎使用和测试，有问题或者交流实验过程可以发issue或者加群734912150<br>


#### 案例
使用yolov3-spp训练oxfordhand数据集并剪枝。下载[数据集](http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz),解压到data文件夹，运行converter.py，把得到的train.txt和valid.txt路径更新在oxfordhand.data中。通过以下代码分别进行基础训练和稀疏训练：<br>
`python train.py --cfg cfg/yolov3-spp-hand.cfg --data data/oxfordhand.data --weights weights/yolov3-spp.weights --batch-size 20 --epochs 100`<br>
<br>
`python -c "from models import *; convert('cfg/yolov3.cfg', 'weights/last.pt')"`<br>
`python train.py --cfg cfg/yolov3-spp-hand.cfg --data data/oxfordhand.data --weights weights/converted.weights --batch-size 20 --epochs 300 -sr --s 0.001 --prune 1`<br>
<br>
训练的情况如下图，蓝色线是基础训练，红色线是稀疏训练。其中基础训练跑了100个epoch，后半段已经出现了过拟合，最终得到的baseline模型mAP为0.84;稀疏训练以s0.001跑了300个epoch，选择的稀疏类型为prune 1全局稀疏，为包括shortcut的剪枝做准备，并且在总epochs的0.7和0.9阶段进行了Gmma为0.1的学习率衰减，稀疏过程中模型精度起伏较大，在学习率降低后精度出现了回升，最终稀疏模型mAP 0.797。<br>
![baseline_and_sparse](https://github.com/tanluren/yolov3-channel-and-layer-pruning/blob/master/data/img/baseline_and_sparse.jpg)
<br>
再来看看bn的稀疏情况，代码使用tensorboard记录了参与稀疏的bn层的Gmma权重变化，下图左边看到正常训练时Gmma总体上分布在1附近类似正态分布，右边可以看到稀疏过程Gmma大部分逐渐被压到接近0，接近0的通道其输出值近似于常量，可以将其剪掉。<br>
![bn](https://github.com/tanluren/yolov3-channel-and-layer-pruning/blob/master/data/img/bn.jpg)
<br>
这时候便可以进行剪枝，这里例子使用layer_channel_prune.py同时进行剪通道和剪层，这个脚本融合了slim_prune剪通道策略和layer_prune剪层策略。Global perent剪通道的全局比例为0.93，layer keep每层最低保持通道数比例为0.01，shortcuts剪了16个，相当于剪了48个层(32个CBL，16个shortcut)；下图结果可以看到剪通道后模型掉了一个点，而大小从239M压缩到5.2M，剪层后mAP掉到0.53，大小压缩到4.6M，模型参数减少了98%，推理速度也从16毫秒减到6毫秒（tesla p100测试结果）。<br>
`python layer_channel_prune.py --cfg cfg/yolov3-spp-hand.cfg --data data/oxfordhand.data --weights weights/last.pt --global_percent 0.93 --layer_keep 0.01 --shortcuts 16`<br>
<br>
![prune9316](https://github.com/tanluren/yolov3-channel-and-layer-pruning/blob/master/data/img/prune9316.png)
<br>
鉴于模型精度出现了下跌，我们来进行微调，下面是微调50个epoch的结果，精度恢复到了0.793，bn也开始呈正态分布，这个结果相对于baseline掉了几个点，但是模型大幅压缩减少了资源占用，提高了运行速度。如果想提高精度，可以尝试降低剪枝率，比如这里只剪10个shortcut的话，同样微调50epoch精度可以回到0.81；而想追求速度的话，这里有个极端例子，全局剪0.95，层剪掉54个，模型压缩到了2.8M，推理时间降到5毫秒，而mAP降到了0，但是微调50epoch后依然回到了0.75。<br>
<br>
`python train.py --cfg cfg/prune_16_shortcut_prune_0.93_keep_0.01_yolov3-spp-hand.cfg --data data/oxfordhand.data --weights weights/prune_16_shortcut_prune_0.93_keep_0.01_last.weights --batch-size 52 --epochs 50`<br>
![finetune_and_bn](https://github.com/tanluren/yolov3-channel-and-layer-pruning/blob/master/data/img/finetune_and_bn.jpg)<br>
可以猜测，剪枝得到的cfg是针对该数据集相对合理的结构，而保留的权重可以让模型快速训练接近这个结构的能力上限，这个过程类似于一种有限范围的结构搜索。而不同的训练策略，稀疏策略，剪枝策略会得到不同的结果，相信即使是这个例子也可以进一步压缩并保持良好精度。yolov3有众多优化项目和工程项目，可以利用这个剪枝得到的cfg和weights放到其他项目中做进一步优化和应用。<br>
[这里](https://pan.baidu.com/s/1APUfwO4L69u28Wt9gFNAYw)分享了这个例子的权重和cfg，包括baseline，稀疏，不同剪枝设置后的结果。

## License
Apache 2.0
