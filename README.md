# GraphEmbedding

## DeepWalk

1. 随机游走生成节点序列
2. 利用节点序列训练Word2Vec
## LINE

1. 接近性定义
   1. 1st: 连接边权重较大
   2. 2nd: 共享较多的相同邻居
2. 嵌入训练
	1. 一阶接近性, 每个节点训练一个自身的embedding, 根据负采样损失进行优化
	2. 二阶接近性, 每个节点有一个自身embedding和上下文embedding, 根据负采样损失进行优化
3. 训练优化
	1. 采样时为了处理边权重不同的情况, 根据Alias方法对边权值较大的节点分配更大的采样权重

## Node2Vec

1. 结合BFS和DFS的随机游走生成序列
   1. $1 / p$的概率回到上一个节点
   2. $1$的概率下一个节点与上一个节点有边
   3. $1 / q$的概率下一个节点与上一个节点无边

2. 利用节点序列训练Word2Vec

## EGES

1. 基本思路与n2v与dw相同
2. 采用weighted randomwalk, 基于边权重进行随机游走, 走到下一个节点概率为:

$$p_{v,w} = \sum_{k\in N(v)}\frac{w_{v,w}}{w_{v,k}}$$

3. 根据用户的访问序列, 以时间/滑窗长度进行session的切分, 原论文中为1小时, 本仓库中的复现取5的window size滑窗, 每次共现即在两个item间加一条边
4. 每个输入点有一个feature, 每个feature分配一个embedding, 最后的item embedding(即W2V)中的U是通过feature embedding与node embedding加权得到, V不变。
5. 原论文采用negative sampled softmax, 本仓库实现采用负采样损失(有空再修改)。
# Content

|Model|Journal|Year|Paper|State|
|:-:|:-:|:-:|:-:|:-:|
|DeepWalk|KDD|2014|[DeepWalk: Online Learning of Social Representations](https://dl.acm.org/doi/abs/10.1145/2623330.2623732)|☑️|
|LINE|WWW|2015|[LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)|☑️|
|Node2Vec|KDD|2016|[node2vec: Scalable feature learning for networks](https://dl.acm.org/doi/abs/10.1145/2939672.2939754)|☑️|
|EGES|KDD|2018|[Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1803.02349.pdf)|☑️|
|GraphSage|||||
|PinSage|||||
|LightGCN|||||
|NGCF|||||
|SDNE|||||
|Struct2Vec|||||