# math
most of time math is a tool to...
describe a process
calculate amount

calculus for optimization
linear algebra for operation
probability for uncertainty inference





## calculus
study of change
assuming limits, continuity, then

- derivative
analyzing function: if a< c< b and f is differentiable on [a,b]
	mean value theorem: f'(c) = (f(b)-f(a))/(b-a)
	extreme value theorem: f have a extreme value on [a,b] when f'(c) == 0 or c == a or b (same as f'(c) not defined), c is called _criticle point_
	increasing/decreasing: when f' == 0, f change its ins/des
	convacity: when f'' == 0, f change its convacity, the point is called _reflection point_
	if f' == 0  when f'' <0, it is a maxima, and vice versa

- integral
differential equation: f'' + f' + f = 0
parametric equation: x=f(t) y=f(t), t be the parameter
series


### differential equation
`y'' + y' + y = 0`
`(d^2 y)/dx^2 + dy/dx + y = 0`
solution is a set of functions
	while algebraic equation has solutions of a set of numbers
if x and y are seperable, seperate them and integrate on both side, use initial condition if exist

euler's method for approximation


#### partial derivative
given f(x), derive y =>
d(f(x))/dy 
= d(f(x))/dx * dx/dy
= f'(x) * dx/dy

#### exponential function
P as population and k is rate of increase 
dP/dt = kP 
1/P * dP = k / dt
ln(P) = kt + c
P = e^(kt+c) = ce^(kt)

#### logistic function
dP/dt = kP(1-N/P), N is the population limit
with paritial derivative
P(t) = NP0/((N-P0)exp(-kt)+P0)
the __LOGISTIC FUNCTION__

#### 2nd order linear differential equation
1st order: f'
2nd order: f''
linear 2nd order: a(x)y'' + b(x)y' + c(x)y = d(x), function a, b, c, d are all linear
_homogenuous_ linear 2nd order: d(x) == 0
homogenuous here means _alike_, the linear combination of solutions are also solutions
if g(x) and h(x) is solution: kg+mh are all solutions

#### translate differential equation to algebraic equation
assume y = exp(kx), so translate y'' to k^2 exp(rx) etc. , called _characteristic_ equation
solve equation as a 2nd order quation, involve two initial conditions for two coeff
if un-homogenuous, guess solutions with _undetermined coefficient_ and validate using the equation


#### laplace transform
a robust tool to solve complex differential equation

transform function of time f(t) to function of frequency f(s)
L(f(t)) = integral(0 ~ INF) (exp(-st)f(t)dt)
L(exp(kt)) == 1/(s-a)
L(sin(at)) == a/(s^2+a^2)
L(t^n) == n!/s^(n+1)
* linear operator
L(kf(t)) == kL(f(t))
* turn derivative to multiplication (like chara equation)
L(f'(t)) == sxL(f(t)) - f(0)
* unit step function
u_c(x) = 1 if x>=c else 0
L(u_c(t)f(t-c)) == exp(-pi s) * L(f(t))
L(u_pi(t)sin(t-pi)) == exp(-pi s)/(s^2+1)

* dirac delta function
y = INF if x == 0 else 0 and
integral(-INF ~ INF)(y) = 1
y_c(t) = 1/2c if -c< t< t else 0

### parametric equation
x=f(t) y=f(t), t be the parameter
derivative dy/dx = (dy/dt)/(dx/dt)
2nd derivative d2y/dx2 = (df'/dt)/(dx/dt), where f' = dy/dx

### series
converge vs. diverge
	test: limit, comparison, alternating, ratio, 
	p series: 1/x^(p) where p is positive, converge only p > 1, diverge if p <= 1
	integral p series with p=1 get ln(x), which is the slowest divergent function
	harmonic series: 1/x^(p) where p is negative, converge if p>= 1

* taylor series 
any given function f to polynomial function p
构造一个多项式p(x), 去拟合任何函数f在0周围的值f(0),此时f(0)应理解为常数
we want p(0) = f(0), p'(0) = f'(0), ...
我们使多项式p(x)的值,一阶导数,高阶导数都等于f(0)对应的常数
p(x)就会等于0次项到高次项的和, 系数为f的对应导数的常数除以微分带来的系数
so we construct p as p(x) = f(0) + f'(0)x + 1/2(f''(0))x^2 +...+ f('n)(0) * x^n/n!
it is _maclaurin series_, a special case of taylor series when x == 0
P_c(x) means P is centered at c instead of 0
P_c(x) = f(c) + f'(c)(x-c) + ...


* legrange error function
E(x) = f(x) - p(x) <= (Mx^(n+1))/(n+1)!, where
n is number of tayler expansion orderm 
M is bound of f(' * n)(x), ex. M for sin(x) is , for exp(x) is exp(x)

* power series
f(x) = sigma(k_n(x-c)^n)
geometry is a special case where all k_n is same and c = 0
if -1< x<1 (convergent range), function is convergent, and can express as __k/(1-x)__, where x is the common ratio
for power series, it is convergence radius of 1, and range is (c-1, c+1)
for special case, integral of geometry series is power series

### multi-variable calculus
calculus with more than one variables
f(x1,x2,...,x10) = ax1 + bx2 - one output, or (ax1, bx2) - two output
or express as matrix[ax1\\bx2] \(\\ for verticle)
think of it on 2d cordinate plane 
output as a new dimension as height, derivative as gradient, change of height, integral as cumulation of change
parameter functions can be shown as a curve(one parameter) or a surface(two parameters) with change of the parameter
vector field and transformation can map one (x,y) parameter-pair to a new one

#### partial derivative
df(x,y)/dx
treat all y in f as a constant
d2f/dxdy
* directional derivative
the derivative on a given vector's direction, [a* df/dx \\ b* df/dy], or vector [a\\b] doc [del(f(x,y))] the gradient
the magnitude of the derivative is the rate of change in its direction
the scale of vector do affect scale of the derivative
so the max rate of change is achieved when direction aligns the gradient

- gradient
is vector combines two derivative: del(f(x,y)) = [df/dx\\df/dy]
	its direction is the steepest ascend, perpendicular to contour line
	its magnitude is the rate of the ascend
gradient always one dimension lower than the model

- chain rule
given f(x,y), and x(t), y(t), the chain rule is essentially
how change of t results in change of x and y and ultimately the change of f
df(x,y)/dt = df/dx * dx/dt + df/dy * dy/dt
dt * rate of change dx/dt = dx, same as dx * df/dx = df, and d(x,y) = dx + dy
if we take (x,y) as vector v(t) = [x\\y], the f(x,y) can be seen as f(v(t))
the chain rule becomes f'(v) dot v'(t), where
f'(v) is [df/dx \\ df/dy], which is the _gradient_
v'(t) is [dx/dt \\ dy/dt]
if we take it into directional derivative's form, it is a derivative in the direction of v'(t), which is the change of v, and the derivative reflects its change in final f

- tangent plane
a linear plane pass one point on the graph, and all lines from the plane is tangent to the graph, so intersection lines of x and y plane is tangent line to the graph on x and y plane
given function f(x,y) = z and point (x0, y0, z0) on graph, L(x,y) = a(x-x0) + b(y-y0) - z0, where
a = df/dx(x0, y0), b = df/dy(x0, y0) (a and b be the slope of x and y plane at point (x0, y0))
in vector form, with x and b be X = [x\\y]:
L(X) = del(f(X0)) dot [X-X0] + X0, where del be the gradient, X0 be a given constant point
the tangent plane can also used as _local linear approximation_, like in newton etc.

* quadratic approximation
despite linear approx., x^2, xy, and y^2 are also included, similar to _tayler expansion_
Q(X) =  X0 + del(f(X0)) dot [X-X0] + 1/2 [X-X0].t @ hessian(f(X0)) @ [X-X0]

- hessian matrix
hessian matrix is a nxn matrix that express the 2nd derivative part of qudaratic approximation
H = [d2f/dx2\\d2f/dxdy, d2f/dydx\\d2f/dy2]

* quadratic form
equations that all terms are in quadratic: ax2 + 2bxy + cy2, etc, they can be written in form
v.t @ W @ v, where W = [a,b\\b,c], v=[x\\y]

### optimization
maxima/minima: where tangent plain of the point (x0, y0) is flat
df(x0, y0)/dx == 0 and df(x0, y0)/dy == 0, or del(f) == [0...0]

- saddle point
max for one dimension and min for the other, while del is [0..0]
just like single variable equation, _concavityy_ can be told by calculating _second order derivative_: f''<0 is max, and vice versa
for multi-variable, it needs d^2f/dx^2, d^2f/dy^2, and d^2f/dxdy, which capture the concavity on diagnol plane
the 2nd derivative saddle test: dxx * dyy - dxy * dxy
if >0: min/max; if < 0: saddle; if == 0: more test
it is 2nd order tayler expansion: 1/2[X-X0].transpose x hessian(f(X0))@[X-X0] \(more on https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/a/reasoning-behind-the-second-partial-derivative-test)

#### constraint optimization
optimize f(x,y) given contraint h(x,y) = c 
the argmax(f) (x0, y0) will be the tangent point of f and h
so the gradient del(f(x0, y0)) will be perpendicular to h(x0, y0), or the other way around
so del(f) and del(h) are in the same direction, the scaler ratio is called __Lagrange multiplier__, written as lambda
and x0, y0, and _L_ can be a equation set using del(f) = L * del(h), and h the constraint equation itself

- Lagrange Equation
translates the constraint optimization of f to an unconstraint optimization (the so-called 'duality')
L(x, y, l for lagrange multiplier lambda) = f(x,y) - l * (h(x,y) - c)
what it essentially do is that no matter what l is, h(x,y)-c will be zero (the given constraint), and the max/min of L is the max/min of f
dL will be zero at it max/min, and to get del(L) == 0, it exactly is the equation set for two same-direction gradient:
dL/dl = h(x,y) - c = 0 (given constraint)
dL/dx = df/dx - l(dh/dx) = 0(same-direction gradient, by x axis)
dL/dy = df/dy - l(dh/dy) = 0(same-direction gradient, by y axis)

- the meaning of Lagrange multiplier
is the sensitivy of max(f) to c, think of g(c) = f_max
given f and h, the L is now fixed and del(L(x, y, l)) is calculated to find max(f)
now think if max(f) as a function of the variable c, c give argmax(f) from del(L(x,y,l)) = 0
the l, _Lagrange multiplier_, is the sensitivity of f_max=g(c) to c, the d(g(c))/dc
observe L = f_max - l(h-c), so dL/dc = l ! and L==f_max at argmax(f)!
think of making two gradient f and h the same direction as the optimization process, and the scalor ratio reveals the sensitivity/rate of change between f and h

#### convex optimization






1. 微积分

1.1 极限

1.2 导数
一阶
二阶

1.2.1 求极值
一阶为0

1.3 泰勒公式
f(x) = f(x0) + f'(x0)dx + 1/2!f''(x0)dx^2 + 1/3!f'''(x0)dx^3 + ...
- exp(x)
if fx = exp(x) and x0=0: exp(x) = 1 + x + ... + 1/n! x^n
- entropy to gini index
信息熵-pln(p) 在p=1的一阶展开,是基尼指数 p(1-p) 
fx = -ln(x), x0=1, f'=-1/x: f(1)+f'(1)(x-1 ) = 1-x
- find square root of a
fx = x^2 - a, 求fx=0时x的值, 假设x0
fx=0 = fx0 + f'x0(x-x0), 带入条件
x-x0 = -fx0/f'x0
x = x0 - f/f'  __newton method__
x = x0 - (x0^2-a)/2x0
我们不停迭代x0=x,当迭代差值够小后停止

# 牛顿
有fx0, 求fx, 用泰勒二阶得到近似Fx
Fx=fx0 + f'x0(x-x0) + 1/2f''x0(x-x0)^2
F'x = f'x0 + f''x0(x-x0), F对x求导,x0的值是常数
F取极值时F'x==0, 所以x = x0-f'x0/f''x0
用二次曲线的极值逼近,对应上面例子中F' = x^2-a=0


1.4 多元导数 (偏导数)
梯度

1.5 凸函数
二阶导数正负判断凹凸
f(x) = x^2, f'' = 2, 凸函数

1.5.1 定义
f(W.t @X) <= W.t @ f(X), where sum(W) is 0

1.5.2 应用
确定凹凸性,上下界,进行不等式替换
信息散度 D(p||q) = sum(p log(p/q))
sum(p) == 1, so D=sum(w@f(x))>=f(sum(wx))
D>= -log(sum(p q/p)) = -log1 = 0


5. 凸优化

## 几何的代数表达
用二维坐标表达点a和b: a(x1,y1), b(x2,y2)
用两点表达连接的直线x: x=wa + (1-w)b, w in R所有实数
直线换成线段, 则w in [0,1]
用方程表达曲线, f(x,y)=0, x和y in R2二维实数空间 
如果y能表达为g(x), 即y不是独立的自变量, 则f(x,y)=y-g(x)
拓展到高维空间, X表示n维向量, f(X)=0表示Rn的超曲面
方便起见, 坐标点也用向量x1,..xn表达


## Affine 仿射
点 a=(x1, y1), b=(x2, y2)
直线 x = wa + (1-w)b, x in R
将点a,b视为向量a,b, 结果即从直线转变为原点到直线上的向量
将二维推广至高维, 结果即从直线转变为超平面
想象三维空间下的平面, 将前两个点视为ab直线上的所有点, 直线和第三点c所连直线即为过三点的平面
就是 __affine set仿射集__ 的概念, 突出"线性相关"

## Convex Set凸集
改变w的定义域, 线段 x = wa + (1-w)b, w in [0,1]
x就是 __convex set凸集__
直觉上理解, 仿射集对应直线, 而凸集对应线段, 仿射集必是凸集, 凸集不一定是仿射集
若有k维, 则有k个点和k个w, 均要满足[0,1]范围
但也可以换一种说法:集合内任意两点a,b间线段的集合为凸集, 即wa + (1-w)b, w in [0,1], 无所谓维度

- Cone 锥
for x in C, w >= 0, sum(wx) always in C, then C is cone
经过原点的射线和锥, 都是锥
半正定方阵的集合是凸锥
https://blog.csdn.net/robert_chen1988/article/details/80479134
A,B in {SPD}, then for any x, xtAx>=0 and xtBx>=0
xt(w1A+w2B)x = w1xtAx+w2xtBx>=0, so w1A+w2B also a SPD, so {SPD} is cone

## hyperplane 超平面
{X | A.t @ x = b}, A, X为向量, b为标量
想象直线a1x1+a2x2=b, 可以看作为(a1,a2).t @ (x1,x2) = b
b=0的话, A和X互相垂直
高维空间中, X为超平面, A为超平面的法线向量
b为X上的向量在A方向上的投影与A点积的标量
- halfspace 半空间
{X | A.t @ x >= b}
大于b则为A同侧的半空间, 反之为A异侧

## Space 空间
- 范数 norm
一个向量的长度度量,记为||x||
||x|| = sqrt(sum(xi^2))
一般默认为二次范数,即欧几里得范数
范数的性质: 非负性, 正定性, 齐次性, 三角不等式
- 范数球
{x| norm(x-xc)<=r^2}, 以xc为圆心,半径为r的球体
如果不是norm2, 比如一维空间, 则x为xc为中心,边长2r的正方形
- 椭球
对欧式球的维度进行拉伸, 乘以对称正定矩阵P, 得到椭球
{x| norm2(x-xc).t P norm2(x-xc) <= r^2}
- 锥
{x,t | norm2(x-xc) <= t}

- 多面体
由半平面组成的交集
多面体都是凸集

### 保持凸性的运算
- 交集运算
- 仿射 f=Ax+b
仿射的和, 笛卡尔积都是凸集
- 透视
P(z,t) = z/t
- 投射
仿射+透视

### 分割超平面
- 支撑超平面
存在法向量a,对C边界上的点x0, a@x0>=a@x(同侧), x为任意C内的点,则x0存在支撑超平面P, P@a=0
凸集边界每一个点都有支撑面
- 分割超平面
若C和D是不想交的凸集, 则存在超平面P, 可以将C和D分离
for all c in C and d in D, a@c<=b and a@+-d>=b, where P@a = b
定义两个集合的距离,为集合间元素的最短距离
距离的垂直平分线为分割超平面


## 凸函数
函数上两点间上的任意点都在函数上方, 即上境图 epigraph, {(x,t) | t>=f(x)}
sum(wf(x)) >=f(sum(wx)) , where sum(w)=1
例子:
exp(ax)
x^a, a>=1 or a<=0
-logx
xlogx
||x|| 范数
max(xi)

- 凸函数的微分性质
一阶: f(x+dx) >= f(x) + dx * f'(x), 函数曲线在切面上方
其支撑超平面给出了函数取值的下界, 从某一点得到全局下界
二阶: 
f(x) = x.t @ A @ x, 
f' = 2Ax
f'' = 2A, 即hessian
若A为对称半正定矩阵,则f为凸

## Jensen不等式
若f是凸函数:
两点: f( w@x1+(1-w)@x2 ) <= w@f(x1) + (1-w)@f(x2)
凸函数的基本定义, 画一个x^2的图像, 选择x1, x2, 两点线段上的点是等式右边, 两点间曲线上的点则是等式左边
多点:sum(w@f(x)) >= f(sum(wx)) , where sum(w)=1
连续: 求和换成积分, int(w dw)=1
int(w@f(x)@dw) >=f(int(wx@dw))
概率化: 把参数w看过是事件x的概率p(x),  
期望 E(f(x)) >= f(E(x))

应用广泛, 比如证明divergence >= 0
D(p||q) = sum(p log(p/q)) = E(log(p/q)), log内取倒数外面加负号
-E(log(q/p)) >= -log(E(q/p)) = -log(sum(p q/p)) = -log(1)=0

## 凸函数的保凸算子
gx = sum(wfx) 加和
gx = f(Ax+b) 仿射
gx = max(fx,..) 求最大/上确界
连续版本的求最大就是: 上确界 sup/inf(fx)
上确界保凸, 下确界保凹

- 求最大可以用几何直观感受
画两条相交的直线f和g, 交点两侧各取一点a和b
假设f左边高, g右边高, 则max(f,g)就是f左边和g右边组成的角, 自然是凸函数
用jensen可以证明, 左侧f上的点和右侧g上的点之间的线段上的任一点要大于f或者g上对应点的值, 符合凸函数定义

- inf and sup 上下确界
离散的sum对应连续的integral
离散的min/max对应连续的inf/sup
sup/inf_x f(x,y) 相当于y为自变量,x被动取argmax(fxy)

# 共轭 conjugation
f"(y) = sup_x( xy-fx )
用二维平面举例, 给定凸函数fx
f"(y)是关于y的函数, 对每个y值来说
xy即为过0点斜率为y的直线
sup( xy-fx )即为这条线与函数fx最远的距离
因为fx是凸函数, 所以最远距离在以y为切线斜率的点获得, 即f'x=y
求
- 例子
fx = 1/2 x^2
f"(y) = sup(xy - x^2/2)
f'x = x = y, 所以x的取值argsup(xy-fx)=y
f"(y) = y^2 - y^2/2 = y^2/2

一个凸函数的共轭函数的共轭函数 是它本身

- fenchel不等式
共轭函数去掉sup: f"y >= xy - fx
调整一下, fx + f"y >= xy
对上面的例子来说, x^2+y^2>=2xy

- 对偶
凸共轭求取关于x的上界限sup_x,对偶函数利用共轭性质通过对偶函数的上界sup_l逼近原函数的下界inf_x

# 拉格朗日
优化问题: 求解方程的最小值 min_x( f(x) )
同时可能含有不等式约束 g(x)<0, 或者等式约束 h(x)=0
要求f,g皆为凸函数, h为线性函数(仿射函数)
拉格朗日方程是优化问题的常用方法, 通过对原问题构造拉格朗日乘子lambda, 将约束条件和原求解方程合并, 形成新的方程组

- 等式约束
n个未知数, k个等式约束, 每个等式约束都有一个乘子l, 所以总共n+k个未知数
L(x,l) = f(x) + l@h(x)
因为f是凸函数, 其极值点处梯度必然与h(x)的梯度相反
想象一下二维平面一个圆形等高线图, 任意画一条直线, 直线所经过的最低点必然与园相切
则此时有拉格朗日方程L的梯度为0, 从而求得lambda和原变量x的关系, 再通过原方程的等式约束, 求得最优值f"(x)和最优解x"
f(x,y) = x^2 + y^2, x+y=100
L(x,y,l) = f(x,y) + l(x+y-100)
dL/dx = 2x+l=0
dL/dy = 2y+l=0
x+y=100
求解上面三组, 得到x=50, y=50, l=100

- 不等式约束
设不等式约束为g(x)
类似等式约束, 构造L(x,l) = f(x) + l@g(x)
跳过推论,从结果上讲, f(x)在取得符合定义域的极值时, L对x的梯度依旧等于0, dL/dx=0
再带入L对l的梯度dL/dl=0, 从而求得最优解, 但是这个过程是艰辛曲折的

# 对偶

## 对偶思想
https://www.zhihu.com/question/27471863
https://blog.csdn.net/mrdonghe/article/details/102713462
原始问题: min(C@X), s.t. A@X>B
对偶问题: max(B@Y), s.t. A@Y<C 

- 例子
从利润最大化下商品数量的分配, 限制条件为各种成本的总数量
对偶到成本最小下原料数量的"稀缺性", 限制条件为每样成本的最低价值

桌子利润10元, 需要5单位木头和3单位人工, 数量为a
椅子利润3元,  需要2单位木头和1单位人工, 数量为b
总共200单位木头, 100单位人工
原问题: max(10a+3b), s.t. 5a+2b<200, 3a+1b<100
对偶问题: 假设单位木头的价值为x, 单位人工价值为y
最小化成本 min(200x+100y), 
最低成本值 s.t. 5x+3y>10, 2x+1y>3

- 互补松弛 Complementary Slackness
当单位木头的价值为0时, 木头不是稀缺资源, 原问题中对木头的限制没有达到
当木头价值非0时, 木头就是稀缺资源, 原问题中对木头的限制在边界上
其价值数值反应的是单位木头的边际价值, 即增加一个木头, 可以在产出上带来的提高

- 与拉格朗日的联系
以上为线性方程组, 拉格朗日方程为其泛化
都是将原问题(拉格朗日方程)的最小值转化为对偶问题的最大值进行求解
影子价格即为乘子l, 互补松弛进一步成为KKT条件

- 与凸优化的联系
凸优化则再多一步, 先把原问题(min(f(x))) 转化为 拉格朗日方程方程 min(L(x,l))
实质是将min(L(x,l)), 先通过凸函数的性质, 转化为min_x( max_l( L(x,l) ) )
再通过对偶转化为 max_l( min_x( L(x,l) ) ), 从所有下界中找一个最大的下界来逼近真正的下界
并得到二者可取等的KKT条件
我们先从极大极小嵌套函数的意义讲起

## 极大极小与极小极大
用广泛情况说明: 给定方程f(x,y), 此时自变量分别为x和y
做极小值运算min_x(f(x,y)), 其意义是给定任意y值, 运算根据y调整x, 使f(x,y)值最小化, 即x=argmin_x(f(x,y))
将其写作g(y)=min_x(f(x,y)), 此时g的自变量为一个y
不妨将y看做是离散变量, 每个yi会由运算min_x分配一个对应的xi, 极小化f使其取值为f(xi,yi)
给定y的极小值g"(y)对应了给定y值下f(x,y)的极小值, 且自动有x=argmin_x(f(x,y))
同理做极大值运算h(x)=max_y(f(x,y)), h"(x)对应了给定x值下f(x,y)的极大值, 且自动有y=argmax_y(f(x,y))
以下将g"(y)省略记做g"y

- 凤尾大于鸡头
在同时给定x,y的情况下, g"y<=f(x,y), 因为g"y是特定y下f能取得的极小值, x也是极小值对应的取值, 一定比任意给定的x更小或相等
同理, f(x,y) <= h"x, h"x是特定x下f能取得的极大值, 
两个式子对所有(x,y)成立, 所以可以把他们连起来得到 g"y <= f(x,y) <= h"x
则 g(y) <= f(x,y) <= h(x), 且其对所有(x,y)成立
对g和h再次求自变量的极值, 因为小项的最大值还是小于大项的最小值 
则 max_y(g(y)) <= min_x(h(x))
则 max_y(min_x(f(x,y))) <= min_x(max_y(f(x,y)))
参考https://blog.csdn.net/q__y__L/article/details/50342931
想象x和y皆为离散, 左边maxmin意味着先计算出f在所有y情况下的一堆极小值, 再从中挑出一个最大的, 即为鸡头, 右边则是凤尾, 鸡为极小值, 凤为极大值

## 从原问题到拉格朗日 Primal
给定凸函数f(x), 凸函数限制 g(x)<0, 目标是求解x"=argmin_x(f(x))同时满足限制
构造拉格朗日方程 L(x,l) = f(x) + l@g(x)
为了让x"也成为L的最优解, 我们定义l = argmax_l( L(x,l) ), 且l>=0
原优化问题min_x(fx)转化为 min_x( max_l( L(x,l) ) ), 对于每个自变量x, max_l会自动调整使L值最大
如果gx>0, 内层max_l会使l取+INF, 从而l@gx=+INF, 外层min_x运算时则会排除这个取值, 从而避免了gx>0的解
如果gx<0, 内层max_l会使l取0, 从而lgx=0, 此时L" = f"x, x取到相同的极值
如果gx=0, 则l取到表示该约束敏感度的值, 类似线性约束中的影子价格
以上两条也是KKT条件表达的互补松弛 

以上便是要求解的拉格朗日方程, p(x) = max_l( L(x,l) ), 一般记做Primal以示和对偶区别, 
p(x)对凸函数求最大值, 依旧是关于x的凸函数, 要求其最小值p"

## 拉格朗日到对偶 Dual
对偶函数就用到了之前的极大极小对调的运算
为了取得p", 我们先构造p(x)的对偶函数d(l), 用d(l)的最大值d"去逼近p"
d(l) = min_x( L(x,l) ), 关于l的线性函数求下确界, 是一个凹函数
d" = max_l( d(l) ) = max_l( min_x( L(x,l) ) ), 就是min和max对调的原拉格朗日方程
运用鸡头凤尾理论, d"是先鸡后头, 所以有d"<=p", d"是原问题下限的逼近
当有KKT的互补松弛时, 二者可取等

## 取等的条件
当以下条件都满足时, min_b(max_a(ab))和max_a(min_b(ab))可取等
1. ab=0
2. a>=0
3. b<=0

- 强对偶条件 KKT
对于凸函数, 当d" == p"时,称为强对偶, KKT条件成立 
1. l@g(x)= 0
2. l    >= 0
3. g(x) <= 0
再加上一个L的梯度为0
4. so df/dx = -l@dg/dx


## 直觉理解
- 为什么构造l@gx?
限制条件为gx<=0, 我们把它看做数字b, 要求b<=0
这等价于min_b(max_a(ab)),a>=0, 这听起来很疯狂但是确实是这么解的
ab有两个变量, 其中不再对b做限制, 而是对a>=0做限制, 否则在max_a时a取负数可以让ab无求大 
max_a(ab)只有一个变量b, 因为max_a运算会自动匹配a=argmax(f(a,b))
同理, min_b(max_a(ab))就只是一个数字, b会自动匹配到使h最小化的b的取值
先看max_a(ab), a根据b的取值分情况讨论
b= 1 >> a=+INF, ab=+INF
b=-1 >> a=0,    ab=0
再看min_b(max_a(ab)), 即从所有b的取值中调出ab最小的值, 自然b=-1, 成功等价于b<=0的要求

- 对谁的偶
是min_b(max_a(ab))的偶
写作max_a(min_b(ab))
a=1 >> b=-INF, ab=-INF
a=0 >> b=-1,   ab=0
可见二者ab取值是相同的, 在某些特定情况下

- 几何直觉之等高线与KKT
做原函数f(x)的等高线, 和不等式现值函数g(x)<=0, g(x)相交的区域为可行域
当我们说在最有点x"处, 某些限制g(x")正在起作用, 说明x"正好骑在g(x)曲线上
此时g(x")=0, 而此时他们的系数l>0
此时原函数f(x)的最优值也正好刚刚接触到可行域, 其梯度df/dx也正好相反于接触到的限制函数g(x)们的梯度之和dg/dx
所以这些g(x)的系数l>0的 
而不起作用的约束, 他的梯度就没有反映, 所以其g(x")<0, 其l=0
以上可以帮助理解若干KKT
df/dx+dg/dx=0
l@g(x)=0
l>=0

https://www.zhihu.com/question/58584814/answer/159863739

- 几何直觉之皮筋与极大极小值
两张图, convex_opt_dual.png
第一张是x vs. L(x,l), 若干不同l取值的L曲线的上确界代表了max_l(L(x,l)), 其中最低点即为x"
第二张是l vs. min_x(L(x,l)), 一条在l取不同值时L(x,l)在全x范围上的极小值

我们首先在第一张图上画出不等式约束g(x), 然后画出一个单调递增的凸函数, 这样就构造了一个在g(x)与y轴的左交点出取得最优值的凸函数f(x)
之后我们画出f(x)加上l由小到大的l@g(x)之后的曲线, 即L(x,l)
形象地看, 这就好比原始f(x)是一个皮筋, 在x=x"最优值处固定不动(因为g(x")=0)
随着l的增大, x"左侧的皮筋向上弯曲(此侧的g(x)>0), 另一侧向下崩直到g(x)作为凸函数再次大于零
画出若干条不同l大小的L(x,l), max_l(L(x,l))就是x"左右两侧最上面的线段连起来的凸函数
观察到当l足够大的时候, 左侧总能超过f(x"), 因为此时l@g(x)中的g(x)>0
当l为0的时候, 右侧就是大于f(x")的
所以这条上确界的最低点, 就是x", 即min_x(max_l(L(x,l)))=x"
我们绘制图片的顺序, 也正是先max_l(上确界线), 在min_x(最低点)的顺序

第二张图中没有直接显示x的信息, 需要我们逐l推理得到
首先从l=0开始, 此时的min_x(L(x,l))为x"左边的低点
随着l的增大, x"左边逐渐上扬, 记录到的min_x(L(x,l))也逐渐上扬
直到左边上扬到在x"处水平, 此时min_x(L(x,l))取到对应的L(x",l), 即f(x")因为x"点不会动因为g(x")=0
接下来min_x(L(x,l))的最小值就会由x"右边的点代替, 因为l@g(x)中的g(x)为负, 当l足够大的时候, 右边原本上扬的曲线就会被这一项压下来变成下凹的曲线, 此时记录到的值又会小于f(x")
于是我们就画出了一个最大值为f(x")的凹函数max_l(min_x(L(x,l))) 

- 几何直觉之共轭函数
做图G = (gx, fx), 得到一个图案
https://www.zhihu.com/question/58584814/answer/1119054535
图案在y轴左侧为满足约束条件的解,最佳解p"为左侧图案的最低点
做直线 y=lambda * gx + height 经过全局最低点(gx, min_fx)
初始lambda为0, 即y为一条直线, 之后增加直线斜率直至接触到左侧最低点
此时得到d" = height, 即直线与y轴交点
若函数为凸,则p" == d"

这个过程可以看作是求原函数相关系数l的共轭函数
https://www.zhihu.com/question/48976354
https://www.zhihu.com/question/58584814/answer/1251533307
设 fx, gx=Ax-b, hx=Cx-d
L = fx + (lA+vC)x - lb - vd
inf_x(L) = -sup_x(L), ignoring -lv-vd as just constant
set y = -lA-vC, L = fx - yx, 
-sup_x(L) = -supx(yx-fx) = -f"(y) = -f"(-lA-vC)
inf_x(L) = -f"(-lA-vC) -lb-vd

- 几何直觉之惩罚项
有见过这个说法, 没有深究

## 求解
因为原问题中x取值argmin_x使L达到极值, 所以dL(x,l)/dx == 0, 可求出x与lambda关系, 再带入dL(x,l)/dl==0并求解

## 例子
- 二次函数 quadratic
fx = x^2, gx = x+1 <= 0
L = x2 + lx + l
dL/dx = 2x+l = 0 so x = -l/2
L = -l^2 / 4 + l
dL/dl = -l/2+1 = 0 so l = 2, x = -1
max L = min fx = 1 when x = -1

fx = x^2, gx = 1-x <= 0
L = x2 - lx + l
dL/dx = 2x - l = 0 so x = l/2
L = -l^2 / 4 + l
l = 2, x = 1, max L = min fx = 1

- 最小二乘 least square
fx = x.t@x, hx = Ax-b = 0 (minimize (Xw-y)^2 in linear regression, x as w, A as x, b as y, fx is loss function)
L = x.t@x + v.t@(Ax-b), v是向量化的l
dL/dx = 2x + v.t@A = 0 so x=-1/2A.t@v
L = -1/4 v.t@A@At@v - b.t@v
dL/dv = -1/2AAt@v - b = 0 to find v, then x
x = (AtA).inv@At@b

- 线性规划 linear program
fx = c.t@x gx = A@x-b <= 0
L = c.t@x + l(A@x-b) = (c+A.t@l).t@x - b.t@l
dL/dx = c + A.t@l = 0
L = -b.t@l if c+A.t@l==0 and l>0 else -inf

- SVM
f(w) = 1/2 norm(w)^2 + C sum(epsilon)
gx = y(wx+b) >= 1-epsilon, epsilon>=0


- MLE
f(x) = -l = -sum(ylogh + (1-y)log(1-h))


参考文章
https://zhuanlan.zhihu.com/p/73028673
https://blog.csdn.net/qq_19528953/article/details/88643184

# 继续优化
线性优化LP, simplex, interior-point
凸二次优化CQP, convex quadratic
半正定优化SDP, semidefinite


# 线性回归
线性回归的目标函数是 最小二乘
min((h-y)^2), h = Xw
也可以用正态分布的最大似然解释
y = Xw+e, p(e) = normal(0, var) = c exp(-e^2/c)
L(w) = product(p(y|w,X)) get max when y-Xw == 0
l = c - sum((h-y)^2), 还是最小二乘
dl = (y-h)x
凸函数, 梯度下降, w = w - del_w

逻辑回归的目标函数是 最大似然
h = h(X,w), 0<=h<=1
p(y) = h^y * (1-h)^(1-y) 
L(w) = product(p(y|w,X)) 
l = sum( ylogh+(1-y)log(1-h) )
dl = (y-h)x
凹函数(加个负号变成凸函数), 梯度上升, w = w + del_w
逻辑回归的广义线性: 对数线性
GLM:y不再是wx+std的正态分布, 而是wx+std正态分布的g(z)连接方程
logistic: g(z) = 1/1+exp(-z)

- 局部线性回归 loess
对距离进行加权

# 下降
迭代: 梯度下降
优化学习率
牛顿/拟牛顿


# Intro

## Optimization Problem
an optimization problem is ...
minimize f(x), subject to g(x) < b
x is a vector of optimization variables
f is the objective function
g are constraint functions
b are limits for the constraints
x' is called the optimal, or the solution of the problem

# convex sets
affine set
convex set: line segment is contained in the set
convex combination: w.t @ x, where sum(w) == 0 and all w are non-negative
convex cone: w.t @ x, where x has two terms and w are non-negative




# convex functions

# convex OP

# Lagrangian Duality

# Algorithms



# Affine 仿射
## 定义
given 一元函数f(x)
也许f很复杂,为了简单化计算fx在x=p附近的值,我们用线性(常数k)逼近
`f(x) = f(p) + k * (x-p)`
k是一个常数, 最优值为f'(p), 即f在p点的导数

推广到函数组F,包含n个函数fn,和m个自变量xm
自变量写作的向量X.shape(m,1), 则F(X)为m维空间投影到n维空间
同样的线性逼近, 用矩阵A.shape(m,n)简化计算FX在X=P附近的值 
`F(X) = F(P) + A * (X-P)`
A是一个和F同shape的矩阵

## Jacobian
A是F在P点的导数矩阵, 即为 Jacobian Matrix
A和原函数F的shape相同,n行对应n个函数的一阶导数,m列对应了m维X的偏微分
A_ij = del_Fi / del_xj


## gradient, hessian, jacobian
假设多元数值方程F(x1,x2,...xn), 自变量X为(1,n)向量
F的梯度del_F,也是(1,n)向量, [del_F/del_x1, del_F/del_x2, ... ], 由F对每个xi的偏微分组成
给定P点, delF(P)就是映射向量P到梯度del_F(P),(1,n)映射到(1,n)
此时可以把delF()看做一个(n,n)的映射矩阵
对delF在P点求导, 得到delF的Jacobian, 就是Hessian, 也就是del_del_F

# Convex

## 定义
边界上任意两点连线, 线上所有点都在上境图内部 (上境图是 凸 的)
`f(ax1+(1-a)x2)<=afx1+(1-a)fx2`

## 一阶条件
一阶导判断函数升降, 凸函数是越升越快的函数
每个点的切线, 都会离上境图越来越远 (上境图是 凸 的)
`fx >= fx0 + dfx0 * (x-x0)`
即 `fx-fx0 >= dfx0 * (x-x0)`
通过定义可以得证

- 超平面
切线对应于高维空间, 就变成了超平面H
凸函数空间S中任意向量s在超平面H的法向量a上的投影都不大于H与S接触的点x的向量在a上的投影
`a.t@s <= a.t@x`

## 二阶条件
二阶导判断函数凹凸, 凸函数二阶导为正
代表曲率的Hessian为半正定
二阶展开, `fx = fx0 + dfx0(x-x0) + 1/2(x-x0).t Hx0 (x-x0)`
根据一阶条件, `fx >= fx0 + dfx0 * (x-x0)`, 二式相减
`dx.t@Hx0@dx>=0`,根据半正定矩阵的定义,Hx0为半正定
即Hx0的所有特征值均为非负数


# 求解

## 梯度下降
求解f(x)的极值
dx = -a * f'(x0)
每次沿梯度相反方向往回蹭一点点, 逐渐使f(x)=0

## 牛顿法
最开始是用来求方程解f(x)=0, 而不是极值 min(f(x))
f(x) = fx0 + f'x0 * dx = 0
dx = fx0/f'x0

转换到极值计算, 其实是求解f'(x)=0, 所以要对f二次泰勒展开
f(x) = fx0 + f'x0 * dx + 1/2f''x0 * dx^2, 对dx求导
df(x)/dx = f'x0 + f''x0 dx = 0
dx = f'x0/f''x0, 相当于求实根的式子
图像上, 可以理解为用一个二次曲线拟合切线, 然后用曲线最低点迭代为下一轮的x

转化到多元函数, 如果将xi当成向量X, f就变成变换矩阵F, f'就是F的梯度del_f, f''就是F的Hessian
dX = Hessian.t @ del_F, 需要计算矩阵的逆, 非常复杂

## 拟牛顿法
大意是沿用泰勒二次展开`df(x)/dx = f'x0 + f''x0 * dx`, 左边变成新x的变量f'(x)
`f'(x)-f'x0 = f''x0 dx`
根据上式构造一个计算更简单的矩阵代替Hessian.t
`dx = f''x0.t @ [f'(x)-f'x0]`
`f''x0.t` 即海森矩阵的逆, 写作B
`f'(x)-f'x0` 为梯度差值, 写作y
`dx = B@y`
B通过预定的计算迭代,保证正定性, B = B0 + ()
有DFP, BFGS, L-BFGS等, 看不太懂
https://zhuanlan.zhihu.com/p/37588590


# 对偶 Duality
MSE, LR等都是凸优化问题, 包括了L1,L2形式, 但是SVM并不是
通过对偶条件, 把一个非凸的问题转化成凸性的对偶问题, 进行凸优化计算, 比如SVM

# 受限优化 constrained optimization
- 等式约束
组成向量, 向量为0
- 不等式约束
在不等边界上的时候, 叫做约束活跃active, 迭代受限制


- 线性规划 Linear programming
f(X) = kX
- 二次规划 Quadratic programming
f(X) = 1/2 X.t @ k @ X

## Lagrange Multiplier
- 等式约束
给定目标函数f(X), 等式约束h(X)=0
其拉格朗日方程le(X) = f(X) + lm * h(X)
因为h(X)=0, 所以le(X)= f(X), f(X)极值转化为le(X)极值
极值点上,目标函数的梯度del_f和约束等式的梯度del_h重叠, 或者说两个函数相切
del_f = lm * del_h
则可以得到n个自变量梯度等式, 和k个等式约束, 共n+k个等式
n个自变量和k个lm, 共n+k个未知数, 可以求解

- 不等式约束
KKT对偶
通过求解minmax和maxmin对偶转换, 逼近目标函数的极值
将约束通过lm加入原式, 非活跃限制的LM=0, 活跃限制的LM为正








