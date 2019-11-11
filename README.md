# yolov3-channel-and-layer-pruning
本项目以[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为基础实现，根据论文[Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)原理基于bn层Gmma系数进行通道剪枝，下面引用了几种不同的通道剪枝策略，并对原策略进行了改进，提高了剪枝率和精度；在这些工作基础上，又衍生出了层剪枝，本身通道剪枝已经大大减小了模型参数和计算量，降低了模型对资源的占用，而层剪枝可以进一步减小了计算量，并大大提高了模型推理速度；通过层剪枝和通道剪枝结合，可以压缩模型的深度和宽度，某种意义上实现了针对不同数据集的小模型搜索。<br>
<br>
项目的基本工作流程是，使用yolov3训练自己数据集，达到理想精度后进行稀疏训练，稀疏训练是重中之重，对需要剪枝的层对应的bn gamma系数进行大幅压缩，理想的压缩情况如下图，然后就可以对不重要的通道或者层进行剪枝，剪枝后可以对模型进行微调恢复精度，后续会写篇博客记录一些实验过程及调参经验，在此感谢[行云大佬](https://github.com/zbyuan)的讨论和合作！<br>
<br>
![稀疏](https://github.com/tanluren/yolov3-channel-and-layer-pruning/blob/master/data/img/1.jpg)

#### 基础训练
用yolov3训练自己的数据集，修改cfg，配置好data，用yolov3.weights初始化权重。<br>
<br>
`python train.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/yolov3.weights --epochs 200 --batch-size 32`

#### 稀疏训练
scale参数默认0.001，根据数据集，mAP,BN分布调整，数据分布广类别多的，或者稀疏时掉点厉害的适当调小s;-sr用于开启稀疏训练；--prune 0适用于prune.py，--prune 1 适用于其他剪枝策略。稀疏训练就是精度和稀疏度的博弈过程，如何寻找好的策略让稀疏后的模型保持高精度同时实现高稀疏度是值得研究的问题，大的s一般稀疏较快但精度掉的快，小的s一般稀疏较慢但精度掉的慢；配合大学习率会稀疏加快，后期小学习率有助于精度回升。<br>
<br>
`python train.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --epochs 300 --batch-size 32 -sr --s 0.0001 --prune 0`

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
`python slim_prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --global_percent 0.8 --layer_keep 0.1`

#### 层剪枝
这个策略是在之前的通道剪枝策略基础上衍生出来的，针对每一个shortcut层前一个CBL进行评价，对各层的Gmma最高值进行排序，取最小的进行层剪枝。为保证yolov3结构完整，这里每剪一个shortcut结构，会同时剪掉一个shortcut层和它前面的两个卷积层。是的，这里只考虑剪主干中的shortcut模块。但是yolov3中有23处shortcut，剪掉8个shortcut就是剪掉了24个层，剪掉16个shortcut就是剪掉了48个层，总共有69个层的剪层空间；实验中对简单的数据集剪掉了较多shortcut而精度降低很少。<br>
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
