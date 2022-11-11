# Exercises on Variational Inference

### Variational inference for classification / logistic regression models

The logistic regression model is a Bayesian classification model 
$$
\begin{align}
	p(Y, w \vert X) = p(w) \times \prod_{i=1}^{n} p(y_i \vert x_i, w) 
\end{align}
$$
where

- $x_i \in \mathbb{R}^{d}$ are features $X = [x_1,x_2, \ldots, x_n]^{T}$

- $y_i \in \{0,1\}$ are binary class labels $Y = [y_1,y_2, \ldots, y_n]^{T}$

- $p(y_i \vert x_i, w)$ is a Bernoulli distribution with $\sigma(u) = 1/(1+e^{-u})$ and 
  $$
  \begin{align}
  	p(y_i \vert x_i, w) &= \sigma(x_i^{T}w)^{y_i}(1- \sigma(x_i^{T}w))^{1-y_i}
  	\\
  	& =\sigma((2y_i-1)\:x_i^{T}w)
  \end{align}
  $$
- $p(w)$ is a prior distribution on weights
  $$
  \begin{align}
  	p(w) = \mathcal{N}(0, I_d)
  \end{align}
  $$
  
  
  

Bayesian inference in this model means

- computing/approximating the posterior
  $$
  \begin{align}
  	q(w; Y,X) \approx p(w \vert Y, X) = \frac{p(Y, w \vert X) }{p(Y\vert X) }
  \end{align}
  $$

- computing/approximating label predictions for new input features
  $$
  \begin{align}
  	p(y_{\ast} \vert x_{\ast}) 
  	&= \int\!dw\: p(w \vert Y, X) \: p(y_{\ast} \vert x_{\ast}, w)
  	\\
  		&= \int\!dw\: q(w; Y, X) \: p(y_{\ast} \vert x_{\ast}, w) 
  \end{align}
  $$

### Variational inference

Since the true posterior $p(w \vert Y, X)$ cannot be computed we approximate it either by a full $q_{\phi}(w; Y, X)  =\mathcal{N}(w; \mu, \Sigma)$ with  $\phi=\{\mu, \Sigma\}$ or diagonal $q_{\phi}(w; Y, X)  =\mathcal{N}(w; \mu, \mathrm{diag}(\sigma^2))$  $\phi=\{\mu, \sigma^2\}$ multivariate Gaussian distribution by optimising the negative evidence lower bound
$$
\begin{align}
	L(\phi; Y,X) = -\sum_{i=1}^{n}\mathbb{E}_{q_{\phi}}[\log p(y_i \vert x_i, w)] + \mathrm{KL}[q_{\phi}(w;Y,X) \vert\!\vert p(w)].
\end{align}
$$
Here we need to approximate $\mathbb{E}_{q_{\phi}}[\log p(y_i \vert x_i, w)]$ and  the predictive distribution using some numerical tricks. Some useful ones are 
$$
\begin{align}
	\int \!dw\: f(x^{T}w)\: \mathcal{N}(u;\mu, \Sigma) &=
  \int \!du\: f(u)\: \mathcal{N}(u;x^{T}w, x^{T}\Sigma x)
  \\
	\int \!du\: \sigma(u)\: \mathcal{N}(u;\mu, \sigma^2) 
	&\approx 
	\sigma(\frac{1}{\sqrt{1+\pi \sigma^2/8}}\mu)
	\\
	\int \!du\: \sigma(u)\: \mathcal{N}(u;\mu, \sigma^2) 
	&\approx 
	\sum_{k=1}^{K}\sigma(\mu+ \sqrt{2}\sigma \tilde{u}_i) \: \frac{1}{\sqrt{\pi}} w_i,
\end{align}
$$

where $\{\tilde{u}_k, w_k\}_{k=1}^{K}$ are the weights and nodes of the univariate Gauss-Hermite quadrature.

### Questions and tasks

For the logistic regression model detailed above

- generate a dataset for linear classification in 2d $p(y=1) = \sigma(c(x^2 + x^1 -1))$ 

- work out the negative evidence lower bound as a function of $\{\mu, \Sigma\}$ and and $\{\mu, \sigma^2\}$optimise it
  - why is the parameterisation  $\Sigma=LL^{T}$  needed?
  - Is there a better parameterisation for $\Sigma$?

- compute prediction for labels $y_{\ast}$ for input $x_{\ast}$; work out the details , what do we need to compute/return? 
- compare for $q_{\phi}(w; Y, X)  =\mathcal{N}(w; \mu, \Sigma)$ and $q_{\phi}(w; Y, X)  =\mathcal{N}(w; \mu, \mathrm{diag}(\sigma^2))$ 
  - what are the differences in terms of storage and computational complexity
  - compare the predictive results on a test set,
  - plot data and predictive class probability
  -  plot the two distributions as functions of $w$  and compare to $p(w \vert Y,X)$, what can we learn




## Variational auto-encoders for handwritten digit generation 

Variational auto-encoders are unsupervised models that learn to embed and generate new data similar to one in a, i.i.d. dataset $X = [x_1, \ldots, x_n]^{T}$. They are Bayesian models where the distribution of the data is
$$
\begin{align}
	p_{\theta}(X) = \prod_{i=1}^{n} p_{\theta}(x_i \vert z_i)\:p(z_i).
\end{align}
$$
where generally we have 
$$
\begin{align}
	p_{\theta}(x_{i}\vert z_{i}) = \mathcal{N}(x_i; \mathrm{NN}^{\mathrm{dec}}_{\theta_{\mu}}(z), \theta_{\sigma^2} I_d)
	\quad
	\text{and}
	\quad
	p(z_i) = \mathcal{N}(0, I_d).
\end{align}
$$
Training is done via maximum likelihood using variational Bayes with a posterior approximation $q_{\psi}(z_i;x) = \mathcal{N}(z_i;\mathrm{NN}^{\mathrm{enc-mean}}_{\phi_{\mu}}(z_i), \mathrm{NN}^{\mathrm{enc-var}}_{\theta_{\mu}}(z_i) I)$. Here $\mathrm{NN}$ denotes a neural network.

The function to optimise is the negative evidence lower bound
$$
\begin{align}
	L(\theta, \phi; X) = 
	-\sum_i E_{q_{\phi}(z_i ; x_i)}[\log p_{\theta}(x_i \vert x_i)] 
	+ \mathrm{KL}[q_{\phi}(z_i; x_i) \vert\!\vert p(z_i)].
\end{align}
$$

### Questions and tasks

- read the original paper https://arxiv.org/pdf/1312.6114.pdf 
- find an implementation for the MNIST handwritten digit dataset and try it out (https://github.com/Lightning-AI/lightning-bolts/tree/master/pl_bolts/models/autoencoders/basic_vae)
- answer the following questions
  - how is this model different from the logistic-regression one
    - in terms of latent variables
    - in terms of likelihood models
  - what are the differences in term of approximate inference
    - in term of latent variables
    - in terms of parameterisation, what does amortisation mean?
    - can we use amortisation for the logistic regression model?
  - what are the differences in term of training
    - what is the role of re-parameterisation, why does it make computation easier what does it replace? (compare to https://arxiv.org/abs/1401.0118) 
    - in light of what we learned, how would you train the logistic regression model with stochastic batch gradient and reparameterisation?  

