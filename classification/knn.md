KNN算法的欧氏距离表达式：<br/>
<img src="http://latex.codecogs.com/gif.latex?\sqrt{\sum\limits_{i=1}^{n}(x_{i}-y_{i})^2}" />

![](http://latex.codecogs.com/gif.latex?x)、![](http://latex.codecogs.com/gif.latex?y)是两个观测值，下标![](http://latex.codecogs.com/gif.latex?i)表示这是观察值的第![](http://latex.codecogs.com/gif.latex?i)个特征。

如果![](http://latex.codecogs.com/gif.latex?x_i)是一个字符串，那肯定是没法计算距离的，因此需要转换为某种数值形式，这样才能用欧氏距离计算公式计算。
