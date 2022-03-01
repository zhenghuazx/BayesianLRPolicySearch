# Green Simulation Assisted Policy Search RL

## This repo includes 
1. implementation of the model-based policy gradient algorithm with transition model risk; 
2. a biomanufacturing process simulation model class and associate it with MDP transition kernel,

![Image of Yaktocat](https://github.com/zhenghuazx/BayesianLRPolicySearch/blob/master/dataszie100-ni25-i300.png)

Remark: the code was tested under tensorflow==1.14.0 and Keras==2.2.5. It may not support later version of tensorflow and Keras.

The algorithms, simulation method and visualizations used in this package came primarily out of research in Xie Wei's lab at the Northeastern University. If you use Green simulation based policy search (GS-RL) in your research, we would appreciate a citation to the appropriate [paper](https://arxiv.org/abs/2006.09919).
# MDP Calibration

Created: November 8, 2021 2:42 PM
Last Edited Time: November 9, 2021 12:26 PM
Property: regular meeting reference
Type: Weekly Sync

# Recap

Basic Setting:  Suppose the state-transition model is unknown and the computer MDP  is specified by the calibration parameters. Given limited state-action trajectory historical data, we will simultaneously calibrate the process model and search for the optimal policy.

# Value Function

[Value function derivation](https://www.notion.so/Value-function-derivation-b90e3cb0465246af909dda0ab49c911e)

# Policy Gradient (Classic)

Like other policy gradient algorithms, we update the weights approximately in proportion to the gradient of the objective:

$$
\theta\leftarrow\theta-\eta_k\nabla J(\theta)
$$

where the policy gradient theorem (continuous state and action)

$$
\begin{align}\nabla_\theta J(\theta)&=\int_{\mathcal{S}}d^\pi(s)\int_{\mathcal{A}} \nabla_\theta \pi(a|s)\left(Q^\pi(s,a)-b^\pi(s)\right)\mathrm{d}a\mathrm{d}s\\
&=\mathbb{E}_{s\sim d^\pi}\mathbb{E}^\pi[\nabla_\theta\log\pi(a|s)\left(Q^\pi(s,a)-b^\pi(s)\right)]
\end{align}
$$

 In discrete case, the policy gradient can be written as 

$$
\begin{align}\nabla_\theta J(\theta)&=\nabla_\theta\left[\sum_{s\in\mathcal{S}}d^\pi(s)\sum_{a\in\mathcal{A}}\pi(a|s)Q^\pi(s,a)\right]\\
&=\left[\sum_{s\in\mathcal{S}}d^\pi(s)\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s)Q^\pi(s,a)+\pi(a|s)\nabla_\theta Q^\pi(s,a)\right]
\end{align}
$$

where the Q value can be defined as follows ([Degris: 2012](https://www.notion.so/Degris-2012-ac937547d63a4cb2a2135f89fcbad0e2) )

$$
Q^\pi(s,a)=\sum_{s^\prime\in\mathcal{S}}P(s^\prime|s,a)[R(s,a,s^\prime)+\gamma V^{\pi}(s^\prime)]
$$

[Degris: 2012](https://www.notion.so/Degris-2012-ac937547d63a4cb2a2135f89fcbad0e2)  prove in the Off-Policy Policy Gradient (OPPG) theorem that we can ignore the
term $\pi(a|s)\nabla_\theta Q^\pi(s,a)$ without introducing bias for a tabular policy.

Assuming  deterministic reward and reward do not depends on next state, 

$$
\begin{align*}Q^\pi(s,a)&=R(s,a)+\gamma\sum_{s^\prime\in\mathcal{S}}P(s^\prime|s,a)V^\pi(s^\prime)\\
&=R(s,a)+\sum_{s^\prime\in\mathcal{S}}P(s^\prime|s,a)\left(\left(I_{|\mathcal{S}|}-\gamma P^\pi\right)^{-1}R^\pi\right)(s^\prime)
\end{align*}
$$

- If we assume an affine transition function with respect to state and action $s^\prime=As+Ba$ and an affine policy with respect to state $a\sim\pi(a|s)=Cs$, where $C$ is $|\mathcal{A}|\times |\mathcal{S}|$ dimensional matrix. Then we can have $P^\pi=A+BC$ because $s^\prime=As+B\pi(a|s)=(A+BC)s$. Also, we can have
    
    $$
    \begin{align*}Q^\pi(s,a)
    &=R(s,a)+\sum_{s^\prime\in\mathcal{S}}P(s^\prime|s,a)\left(\left(I_{|\mathcal{S}|}-\gamma (A+BC)\right)^{-1}R^\pi\right)(s^\prime)\\
    &=R(s,a)+s^\top(A+BC)^\top\left(I_{|\mathcal{S}|}-\gamma (A+BC)\right)^{-1}R^\pi
    \end{align*}
    $$
    
    Notice that $\theta\equiv C$. Then 
    
    $$
    \nabla_\theta \log\pi(a|s)=\nabla_\theta \log\theta=\left(\frac{1}{\theta_{ij}}\right)_{i\in\mathcal{A},j\in\mathcal{S}}=\text{inv}(\theta)
    $$
    
    where $\text{inv}(\cdot)$ denote the element-wise matrix reverse. Then the policy gradient becomes
    
    $$
    \nabla_\theta J(\theta)
    =\mathbb{E}_{s\sim d^\pi}\mathbb{E}^\pi[\left(Q^\pi(s,a)-b^\pi(s)\right)]\text{inv}(\theta)
    $$
    
    Question: can we derive the stationary distribution $d^\pi(\cdot)$.
    
    Then we can have the policy gradient estimator as 
    
    $$
    \nabla_\theta J(\theta)=\frac{1}{n}\sum^n_{i=1}(Q^\pi(s_i,a_i)-b^\pi(s_i))\text{inv}(\theta)
    $$
    

# State Transition Calibration

Assume that the real observation of next state $s^\prime$ is affine in $s$ and $a$, that is

$$
s^\prime=g(s,a)=As+Ba+\epsilon_g+\epsilon_{p}
$$

Our goal is to reduce the model discrepancy.  We have the  process inherent stochasticity $\epsilon_g\sim\mathcal{N}(0,\Sigma_g)$ and model misspecification (risk) error $\epsilon_p\sim \mathcal{N}(0,\Sigma_p)$. Assume both noises are independent and we have $\sim \mathcal{N}(0,\Sigma)$, where $\Sigma = \Sigma_g+\Sigma_p$. 

To optimize the calibration parameters $A$ and $B$, we can consider that generalized least square error. For each transition $\{(s_i,a_i,s_i^\prime,r_i)\}_{i\leq N}$ *from the real system (MDP), we calibrate the modeled MDP by solving the generalized least square*

$$
\min_{A,B,\Sigma}\Vert A \pmb{s}_i+B\pmb{a}_i- \pmb{s}^\prime_i\Vert_{\Sigma}=\left( A \pmb{s}_i+B\pmb{a}_i- \pmb{s}^\prime_i\right)^\top\Sigma \left( A \pmb{s}_i+B\pmb{a}_i- \pmb{s}^\prime_i\right).
$$
