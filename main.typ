#import "@preview/codly:1.1.1": *
#show: codly-init

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "fr"
)

#set page(
  header: context {
    if counter(page).get().first() == 1 [
      Decaluwé Maxime (35942200)
      #h(1fr)
      Dewell Guerand (84792200)
    ]
  },
  margin: (top: 5%, x:5%, bottom: 5%),
  numbering: "1",
  number-align: center,
)



// Math settings
#set math.mat(delim: "[", gap: .5em)

#v(1em)

#align(center, text(18pt)[
  *LINMA1731 - Stochastic processes: estimation and prediction*\
  Project : Tracking a Lorenz particle
])


#set heading(numbering: "1.1 -")


= Theoretical Questions 

If we have the Lorenz system defined by the following equations:

#align(center , 
$
  cases(dot(x) = sigma(y - x) , 
 dot(y) = x(rho - z) - y,
 dot(z) = x y - beta z)$)

where x, y and z stand for the position coordinates of the point in the system and σ, ρ and β
 are positive real parameters. Lorenz used the values $sigma = 10$ , $rho = 28$ and $beta =  8/3$ for which the
 system shows a chaotic behaviour.

==  Empirical probability density function 

Starting point at $(x,y,z) = (1,1,1)$ at time $t = 0$. The total time is $100 s$ and we make step of time of $0.02 s$. We have the following restriction for the domain of respectively $x,y,z$ : [-20,20] x [-30,30] x [0,50].

a) 
To be able to compute the empirical PDF, we are going to divide the spatial domain into 3D cubic box of length $l = 5$. For exemple the interval $x$ is then going to be : 
$x in $ [-20 ,-15] x [-15,-10] x [-10 , -5] x [-5,0] x [0,5] x [5,10] x [10,15] x [15,20].
We will then have 8 intervals for $x$.
We will then start from the point $(x_0,y_0,z_0) = (1,1,1)$ and for each step of time $ t_k =  k times 0.02 $ with $k = 0,1,2,...,5000.$ We have the point $ (x_k,y_k,z_k) $. 
The next step is computing in wich box is the point and increment the counter of the box.
We will then compute the empirical PDF by dividing the number of points in each box ($N_(i,j,k)$) by the total number of points ($N_("total")$) iid : 
$ P_(i,j,k) = (N_(i,j,k)/N_("total")) $

If we project the empirical PDF on the different plan we obtain the following result :

#grid(
  columns: (1fr, 1fr, 1fr),
  rows: (auto),
  gutter: -2em,
  [
    #figure(
      image("XY.png", width: 100%),
      caption: [Plan XY]
    )
  ],
  [
    #figure(
      image("XZ.png", width: 100%),
      caption: [Plan XZ]
    )
  ],
  [
    #figure(
      image("YZ.png", width: 100%),
      caption: [Plan XY]
    )
  ]
)

What we can see is that there is a majority of point in the middle of the domain. 
In the plan XY it's seems like we have a normal distribution. We can clearly observe the general shape of the system just by looking at the different plan. 

b) 
To mesure the distance between twee empirical PDF, we made the choice to mesure it by using the Kullback-Leibler divergence. The Kullback-Leibler divergence is a measure of how one probability distribution diverges from a second expected probability distribution. The Kullback-Leibler divergence is defined as follows #footnote[https://fr.wikipedia.org/wiki/Divergence_de_Kullback-Leibler] :

For twee discrete probability distributions $P$ and $Q$ defined on the space $X$, the Kullback-Leibler divergence of $P$ from $Q$ is defined as:

$ D_("KL")(P||Q) = sum_(x in X) P(x)log P(x)/Q(x) $

where $P(x)$ and $Q(x)$ are the probabilities of the event $x$ according to the distributions $P$ and $Q$, respectively. 

The second choice of distance is the distance of
Bhattacharyya. The Bhattacharyya distance is a measure of the similarity of two probability distributions. It is defined as the negative logarithm of the Bhattacharyya coefficient, which is defined as follows #footnote[https://fr.wikipedia.org/wiki/Distance_de_Bhattacharyya] : 


For twee discrete probability distributions $P$ and $Q$ defined on the same probability space $X$, the Bhattacharyya distance between $P$ and $Q$ is defined as:

$ D_("B")(P,Q) = -log(sum_(x in X) sqrt(P(x)Q(x))) $

where $P(x)$ and $Q(x)$ are the probabilities of the event $x$ according to the distributions $P$ and $Q$, respectively.

c) 
To mesure the impact of the parameters $sigma , rho , beta$ on the distribution. We have made a function were we can observe the evolution of the system of lorenz by adjusting the paramters. 

Influence of parameter $sigma$: 
- *Small* : The system tend to have a more regular behaviour. After a certain value of $sigma$ the system start to have stable behaviour.

- *Big* : The general behaviour of the system is chaotic. After a certain value of $sigma$ the system tend to have a special behaviour
#grid(
  columns: (1fr, 1fr),
  rows: (auto),
  gutter: -2em,
  [
    #figure(
      image("small_sigma.png", width: 100%),
      caption: [Small $sigma$]
    )
  ],
  [
    #figure(
      image("sigma_grand.png", width: 100%),
      caption: [Big $sigma$]
    )
  ]
)



Influence of parameter $rho$:
- *Small* : 
The system converge to a stable point. The system is not chaotic.
- *Middle* : 
The system is oscillating between two points. The system is not chaotic.

- *Big* : 
The system is chaotic. We can not predict the future position of the system.
#grid(
  columns: (1fr, 1fr , 1fr),
  rows: (auto),
  gutter: -2em,
  [
    #figure(
      image("rho_small.png", width: 100%),
      caption: [Small $rho$]
    )
  ],
  [
    #figure(
      image("rho_mid.png", width: 100%),
      caption: [Middle $rho$]
    )
  ],
  [
    #figure(
      image("rho_big.png", width: 100%),
      caption: [Big $rho$]
    )
  ]
)


Influence of parameter $beta$:
- *Small* :
The system tend to be less chaotic.
- *Big* :
The chaos can be amplified. The system is more chaotic.
#grid(
  columns: (1fr, 1fr),
  rows: (auto),
  gutter: -2em,
  [
    #figure(
      image("betha_small.png", width: 100%),
      caption: [Small $beta$ ]
    )
  ],
  [
    #figure(
      image("betha_big.png", width: 100%),
      caption: [Big $beta$ ]
    )
  ]
)
 

If we mesure the distance between our empirical PDF and a new one calculated with different paramters ($sigma = 5 , rho = 20 , beta = 2$) we can see that the distances are not the same. In term of number we can see that the distance of KL divergence is $18.13$ and the distance of Bhattacharyya is $4.51$. The distance of KL divergence is bigger than the distance of Bhattacharyya.The distance of Bhattacharyya is smaller because she is not impacted as mush as the KL divergence by the presence of 0 in one distribution and not in the other one.As the twee distance are big with respect to the value that they can take, we can say that the two empirical PDF are not really close.

d) 
If we mesure the distance between our empiral PDF and the same one but with an other start point $((x_0 , y_0 ,z_0) = (10 , 10 , 10))$  The distance is bigger for the Kullback-Leibler divergence than for the Bhattacharyya distance. In term of number we have a distance of $0.896$ for the KL Divergence and a distance of $0.0448$ for the Bhattacharyya distance. As the twee distance are small with respect to the value that they can take, we can say that the two empirical PDF are really close.

In conclusion for this part we can see that the initial state does not impact as mush as the paramter of the system. The distance between the two empirical PDF is smaller when we change the initial state than when we change the parameters of the system.



== Particle Filter

If we have a dynamic state space model : 

#align(center ,
$
  cases(x_t = f(x_(t-1) , v_t) , 
 y_t = g(x_t , w_t))$)

where $x_t in RR ,y_t in RR $ and $v_t , w_t$ are the noise. Our goal is to estimate $x_t$ using observations $y_(1:t) = {y_1,y_2,...,y_t}$. Since we don't know exactly $p(x_(0:t)|y_(1:t))$, we are going to approximate it using a set of weighted samples. That bring some problems as over time, some particles have very low weights, while a few dominate. This leads to weight degeneracy, where most particles contribute very little to the estimate. To fix this, we perform resampling, which eliminates low-weight particles and duplicates high-weight ones.
Here we are going to compare the performance of three type of resmapling based on there complexity and variance.We are then going to compute _var_($N_t^(i)$)
- Multinomial Resampling :
As the $N_t^(i)$ are computed from a multinomial distribution with paramters $N$ and $tilde(w)_t^(i)$ we have directly that the variance is given by : _var_($N_t^(i)$) = $N * tilde(w)_t^(i) * (1 - tilde(w)_t^(i))$.
- Residual Resampling : 
The first syep is to compute the deterministic part, so for each particle $i$, we are going to compute a integer $tilde(N)_t^(i) = floor(N tilde(w)_t^(i))$.
By doing this we ensure that particle with a high weight gets a certain number of copies without any randomness.
After having computed the deterministic part, there are still $overline(N)_t = N - sum_(i=1)^N tilde(N)_t^(i)$ particles that don't have a weight. 
For these particles we are going to apply a procedure called SIR (Sampling Importance Resampling) wich give us the following new weights : $w_t^('(i)) = overline(N)^(-1)_t (tilde(w)_t^((i)) N - tilde(N)_t^(i))$
The total number $N_t$ is then given by the sum of the deterministic part and the random part. 
As the deterministic part is deterministic, we have that the variance of the deterministic part is 0.
The variance is then given by the variance of the random part which is given by : _var_$(N_t^((i))) = overline(N)_t w_t^('(i)) (1 - w_t^('(i)))$
By doing some more computation we can see that the variance is smaller than the one of the multinomial resampling : As _var_$(N_t^((i))) = overline(N)_t w_t^('(i)) (1 - w_t^('(i))) = overline(N)_t overline(N)^(-1)_t (tilde(w)_t^((i)) N - tilde(N)_t^(t)) (1 - overline(N)^(-1)_t (tilde(w)_t^((i)) N - tilde(N)_t^(i)))$.

If we compare to the multinomial resampling. 
The term $tilde(w)_t^((i)) N - tilde(N)_i$ is smaller than $N * tilde(w)_t^((i))$ because we remove $tilde(N_t^(i))$. The part $1 - overline(N)^(-1)_t (tilde(w)_t^((i)) N - tilde(N)^(i)_t)$ is also smaller than $1 - tilde(w)_t^((i))$ because we remove $tilde(N_t^(i))$ and we divide by $overline(N)_t^(i)$. So we can conclude that the variance of the residual resampling is smaller than the one of the multinomial resampling.
- Systematic Resampling :
If we define the cumulative sequence :
$ C_i = sum_(j=1)^i tilde(w)_t^((j))$ with $C_0 = 0$ and $C_M = 1$.
We have that the cumulative sums partition the interval $[0,1]$ into M subintervals: 
$(C_0,C_1],(C_1,C_2],...,(C_(M-1),C_M]$. Each of these interval have a length of $tilde(w)_t^((i))$.
As we sample a set $U = {u_1 ,u_2,...,u_N}$ of N points in $[0,1]$, if we count the number of points $N_t^((i))$ iid : $N_t^((i)) = "#"{u_k in U : C_(i-1) < u_k <= C_i}$
In a pure multinomial resampling scheme, the variance in the number of offspring $N_t^((i))$ would be : 
_var_$(N_t^((i))) = N * tilde(w)_t^((i)) (1 - tilde(w)_t^((i)))$.

However, because systematic or stratified sampling forces an even spread of points, the variance is further reduced. For this strategy, the variance becomes
_var_$(N_t^((i))) = overline(N)_t w_t^('(i)) (1 - overline(N)_t w_t^('(i)))$.

As we can see it is smaller than the one of the Systematic resampling as we substract $overline(N)_t w_t^('(i))$ wich is greater than $tilde(w)_t^('(i))$.



#table(
  columns: 3,
  [*Type of sampling*], [*Variance*],[*Time*],
  [Multinomial Resampling], [3.6303],[0.000070s] ,
  [Residual Resampling], [3.1031],[0.000060s],
  [Systematic Resampling], [3.0735],[0.000040s],
  
)
By applying the three differents method of resampling we can see that the systematic resampling is clearly the best one in term of variance and time. 
To compute these data we have used our sample from point 1.2. We can also plot the different repartion of the particles after resampling.
#image(
  "resampling.png",
)

It is much clear on a random set of weights : 
#image(
  "resampling_random.png",
)

