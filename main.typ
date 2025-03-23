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
  margin: (top: 5%, x:10%, bottom: 5%),
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
 are positive real parameters. Lorenz used the values σ = 10, ρ = 28 and β = 8
 3 for which the
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



