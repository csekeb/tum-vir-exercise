# Homework on Variational Inference

This document provides detailed explanations for the models implemented in

- `logistic-regression/model_logreg_mvn.py` a Bayesian logistic regression / classification model for linearly separable data
- `vae/model_vae.py`  an unsupervised model for handwritten digit generation.

The implementations use the `pytorch-lightning` framework and the relevant parts of code are

- `logistic-regression/model_logreg_mvn.py`, functions `forward`, `accuracy` and `loss`as well as `logistic-regression/run.py`
- `vae/model_vae.py`,  functions `forward` and `loss` as well as `vae/run.py` and`vae/config_vae.yaml`.  

### Bayesian logistic regression / classification

The Bayesian logistic regression model is a classification model defined as
$$
\begin{align}
	p(Y, w \vert X) &= p(w) \: \prod_{i=1}^{n} p(y_i \vert x_i, w)
  \\
  & =\mathcal{N}(w; 0, \sigma_{\mathrm{prior}}\: I_d) \: \prod_{i=1}^{n}\mathrm{Bernoulli}(y_i; x_i w )
\end{align}
$$
where

- $x_i \in \mathbb{R}^{d}$ are features $X = [x_1,x_2, \ldots, x_n]^{T}$
- $y_i \in \{0,1\}$ are binary class labels $Y = [y_1,y_2, \ldots, y_n]^{T}$
- $\mathcal{N}(w; 0, \sigma_{\mathrm{prior}}\: I_d)$ denotes a multivariate normal distribution on a weight vector $w \in \mathbb{R}^{d}$
- $\mathrm{Bernoulli}(y_i; x_i w ) = \sigma(x_i w)^y_i (1-\sigma(x_i w))^{1-y_i}$ with $\sigma(x_i w) = 1/(1+e^{-x_i w})$.

Bayesian inference in this model means

- computing/approximating the posterior
  $$
  \begin{align}
  	q(w) \approx p(w \vert Y, X) = \frac{p(Y, w \vert X) }{p(Y\vert X) }
  \end{align}
  $$

- computing/approximating label predictions for new input features
  $$
  \begin{align}
  	p(y_{\ast} \vert x_{\ast}) 
  	&= \int\!dw\: p(w \vert Y, X) \: p(y_{\ast} \vert x_{\ast}, w)
  	\\
  		&= \int\!dw\: q(w) \: p(y_{\ast} \vert x_{\ast}, w) 
  \end{align}
  $$

### Variational inference

Since the Bayesian posterior $p(w \vert Y, X)$ is analytically intractable, we approximate it either by a multivariate Normal $q_{\phi}(w)  =\mathcal{N}(w; \mu, \Sigma)$ with  $\phi=\{\mu, \Sigma\}$ or diagonal/factorised multivariate Normal $q_{\phi}(w)  =\mathcal{N}(w; \mu, \mathrm{diag}(\sigma^2))$  $\phi=\{\mu, \sigma^2\}$ by optimising the negative evidence lower bound (ELBO)
$$
\begin{align}
	L(\phi; Y,X) &= -\sum_{i=1}^{n}\mathbb{E}_{q_{\phi}(w)}[\log p(y_i \vert x_i, w)] + \mathrm{KL}[q_{\phi}(w) \vert\!\vert p(w)].
	\\
	& \geq \log p(Y\vert X) 
\end{align}
$$
Here we need to numerically approximate $\mathbb{E}_{q_{\phi}(w)}[\log p(y_i \vert x_i w)]$ and  the predictive distribution. We can use 
$$
\begin{align}
	\int \!dw\: f(xw)\: \mathcal{N}(w;\mu, \Sigma) &=
  \int \!du\: f(u)\: \mathcal{N}(u;xw, x\Sigma x^{T})
  \\
  &=
  \int \!d\epsilon\: f(x w + \sqrt{x\Sigma x^{T}}\epsilon)\: \mathcal{N}(\epsilon;0,1)
  \\
	\int \!du\: \sigma(u)\: \mathcal{N}(u;\mu, \sigma^2) 
	&\approx 
	\sigma\left(\frac{1}{\sqrt{1+\pi \sigma^2/8}}\mu\right)
	\\
	\int \!du\: \sigma(u)\: \mathcal{N}(u;\mu, \sigma^2) 
	&\approx 
	\sum_{k=1}^{K}\sigma(\mu+ \sqrt{2}\sigma \tilde{u}_i) \: \frac{1}{\sqrt{\pi}} w_i,
\end{align}
$$

where $\{\tilde{u}_k, w_k\}_{k=1}^{K}$ are the weights and nodes of the univariate Gauss-Hermite quadrature.

### Key variational inference concepts in this model / implementation

The approximation of the EBLO objective in implemented in the function.

```
def loss(self, features, labels):
```

please follow the implementation there.

#### Batch learning

For large datasets it is often infeasible to use all training data in each training step, for example, due to memory restrictions, therefore, we use the approximation
$$
\begin{align}
	\sum_{i=1}^{n}\mathbb{E}_{q_{\phi}(w)}[\log p(y_i \vert x_i, w)]  \approx n \frac{1}{\vert S\vert} \sum_{s \in S}\mathbb{E}_{q_{\phi}(w)}[\log p(y_i \vert x_s, w)],
\end{align}
$$
that is, we approximate the objective by using only a random subset $S \subseteq \{1, \ldots, N\}$ to represent the data. This makes the objective stochastic w.r.t. sampling $S$ but with the right optimisation procedure (stochastic gradient descent with decaying learning rates) convergence can still be achieved. 

**Implementation** This concept is implemented via the `DataModuleFromNPZ` in `run.py` which uses data batches of size `size_batch`, see 

```
dm = DataModuleFromNPZ(
    data_dir="data_logistic_regression_2d",
    feature_labels=["inputs", "targets"],
    batch_size=64,
    num_workers=4,
    shuffle_training=False
)
```

in `run.py` and the code line 

```
logp_expct = self.size_data*torch.mean(p_labels.log_prob(labels.repeat((1,self.n_samples_mc))))
```

in function `loss`.  We take the batch average of the expected log likelihood and rescale it with the size of the training data set.

#### Stochastic gradient learning

The expectations $\mathbb{E}_{q_{\phi}(w)}[\log p(y_i \vert x_s, w)]$ can rarely be computed exactly or approximated accurately via quadrature methods---quadrature works well only in 1D.  For this reason we often use Monte-Carlo estimates
$$
\begin{align}
	\mathbb{E}_{q_{\phi}(w)}[\log p(y_i \vert x_s, w)] \approx \frac{1}{ R } \sum_{w^{(r)} \sim q(w)} \log p(y_i \vert x_s, w^{(r)}),
\end{align}
$$
that is, we sample $R$ samples $w^{(r)} \sim q(w)$ and average. This also makes the objective stochastic but we hope that with the right number of samples and the right optimisation procedure the optimisation can still converge (see above). 

**Implementation** This is implemented in line by using 

```
# data distribution via MC samples
p_labels   = Bernoulli(logits=z_samples)
# computing the MC samples based expected log likelihood with batch learning correction
logp_expct = self.size_data*torch.mean(p_labels.log_prob(labels.repeat((1, self.n_samples_mc))))
```

As you can see, in our implementation we do something cleverer by using MC samples from $z=xw$, we explain this  in the next two sections.

#### Reparameterisation of stochastic variables

If a random variable can be represented as a deterministic (preferably differentiable) function of some other/basic random variable with fixed or no parameters, say,
$$
\begin{align}
	z = f_{\theta}(\epsilon), \quad \epsilon\sim p_0(\epsilon), \qquad p_{\theta}(z) = \int\! d\epsilon \: p_{0}(\epsilon)\:\delta(z-f_{\theta}(\epsilon))
\end{align}
$$
then we can rewrite expectations w.r.t $p_{\theta}(s)$ as expectation w.r.t. the base distribution $p_{0}(\epsilon)$. For example, for a function $g()$ we can write and approximate the expectations as 
$$
\begin{align}
	\mathbb{E}_{p(z)}[g(z)] = \mathbb{E}_{p_{0}(\epsilon)}[g(f_{\theta}(\epsilon))] \approx \frac{1}{R}\sum_{\epsilon^{(r)} \sim p_{0}(\epsilon)}g(f_{\theta}(\epsilon^{(r)}).
\end{align}
$$
This makes the MC estimates of expectations easily computable and differentiable w.r.t. $\theta$, that is, 
$$
\partial_{\theta}\mathbb{E}_{p(z)}[g(z)] = \mathbb{E}_{p_{0}(\epsilon)}[\partial g(f_{\theta}(\epsilon)) \partial_{\theta}f_{\theta}(\epsilon)] \approx \frac{1}{R}\sum_{\epsilon_r \sim p_{0}(\epsilon)}\partial g(f_{\theta}(\epsilon)) \partial_{\theta}f_{\theta}(\epsilon).
$$
In our model we have $w =  \mu +  L \epsilon, LL^{T}=\Sigma, \epsilon \sim \mathcal{N}(0, I)$. Hence we can use Monte-Carlo samples from $\epsilon$ to approximate the objective and easily differentiate the approximation. Since we need a positive definite $\Sigma$, and this an $L$ with a positive diagonal,  we use a parameterisation $L = L_{\mathrm{lower}} + \mathrm{diag}(e^{\gamma})$, where $L_{\mathrm{lower}}$ is a strictly lower triangular matrix and $\gamma \in \mathbb{R}^{d}$ .

**Implementation** This is implemented in lines

```
# reparameterisation of stochastic variables
L      = self.weights_chol() 
p_post = MultivariateNormal(loc=self.weights_loc.squeeze(), scale_tril=L)
```

via the helper function

```
def weights_chol(self):
    return torch.tril(self.weights_scale_lower, -1) + torch.diag(torch.exp(self.weights_scale_logdiag))
```

and the parameterisation is defined the `__init__` function in lines

```
self.weights_loc           = nn.Parameter(torch.zeros((self.dim,1)), requires_grad=True) 
self.weights_scale_logdiag = nn.Parameter(torch.zeros((self.dim)), requires_grad=True) 
self.weights_scale_lower   = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True) 
```

#### Local reparameterisation

We observe that the likelihood terms $\mathrm{Bernoulli}(y_i; x_i w )$ depend only on $x_i w$ hence instead to sampling from $w \sim q(w) = \mathcal{N}(\mu, \Sigma)$ we can sample from $z_i=x_iw \sim = \mathcal{N}(x_i\mu, x_i\Sigma x_i^{T})$, that is
$$
\begin{align}
	\mathbb{E}_{q_{\phi}(w)}[\log \mathrm{Bernoulli}(y_i; x_i w )] 
	&= \mathbb{E}_{w \sim \mathcal{N}(\mu, \Sigma)}[\log \mathrm{Bernoulli}(y_i; x_i w )]
	\\
	&=\mathbb{E}_{z_i \sim \mathcal{N}(x_i\mu, x_i\Sigma x_i^{T})}[\log \mathrm{Bernoulli}(y_i; z_i)]
	\\
	&\approx\frac{1}{R} \sum_{z_i^{(r)} \sim \mathcal{N}(x_i\mu, x_i\Sigma x_i^{T})}\log \mathrm{Bernoulli}(y_i; z_i^{(r)})
	\\
	&=\frac{1}{R} \sum_{\epsilon_i^{(r)} \sim \mathcal{N}(0, 1)}\log \mathrm{Bernoulli}(y_i; x_i\mu + \sqrt{x_i\Sigma x_i^{T})} \epsilon^{(r)})
	\\
\end{align}
$$
thus significantly reducing the variance of the stochastic approximation of the objective, the latter step is implemented in `pytorch` in `Normal(loc=z_loc, scale=z_scale).rsample([self.n_samples_mc])`.

**Implementation** Approximating the expectations and by using all the steps detailed above is implemented in

```
# local reparameterisation and MCsampling
z_loc     = torch.matmul(features, self.weights_loc).squeeze()
z_scale   = torch.sqrt(torch.sum(torch.matmul(features, L)**2, dim=-1, keepdim=True)).squeeze()
z_samples = Normal(loc=z_loc, scale=z_scale).rsample([self.n_samples_mc]).transpose(0,1)

# data distribution via MC samples
p_labels   = Bernoulli(logits=z_samples)
# computing the MC samples based expected log likelihood with batch learning correction
logp_expct = self.size_data*torch.mean(p_labels.log_prob(labels.repeat((1, self.n_samples_mc))))
```

### Questions and tasks (at home)

For the logistic regression model detailed above

- Run the code with `python run.py` and check metrics with `tensorboard --logdir lightning_logs` .
- Change `batch_size`, `n_samples_mc`, `max_epochs`, `lr` and the optimisation algorithm, what do you notice?
- Try to implement the diagonal version of `class ModelLogisicRegressionMvn(LightningModule)`, what changes do you have to make? What advantaged does this method have when the number of features $d$ is large?

- Compare for $q_{\phi}(w; Y, X)  =\mathcal{N}(w; \mu, \Sigma)$ and $q_{\phi}(w; Y, X)  =\mathcal{N}(w; \mu, \mathrm{diag}(\sigma^2))$ .
  - What are the differences in terms of storage and computational complexity?
  - Compare the predictive results on a test set.
  - Plot data and predictive class probability.
  -  Plot the two distributions as functions of $w$  and compare to $p(w \vert Y,X)$, what can we learn?




## Unsupervised handwritten digit recognition / variational auto-encoder

Variational auto-encoders are unsupervised models that learn to embed and generate new data similar to one in an i.i.d. dataset $X = [x_1, \ldots, x_n]^{T}$. They are Bayesian models where the distribution of the data is
$$
\begin{align}
	p_{\theta}(X) = \prod_{i=1}^{n} \int\,dz_i \: p_{\theta}(x_i \vert z_i)\:p(z_i).
\end{align}
$$
and learning is done by maximising the likelihood $p_{\theta}(X)$ w.r.t. the model parameters $\theta$. It is hoped that after learning

- samples $z \sim p(z), x\sim p(x \vert z)$  generate date similar to that in $X$
- the Bayesian posteriors $p(z \vert x)$ embed the data in a space where components of $z$ or the new dataset $z \sim p(z_i \vert x_i)$ can tell us more about the data, say, the embedding cluster nicely. 

Where generally we have 
$$
\begin{align}
	p_{\theta}(x_{i}\vert z_{i}) = \mathcal{N}(x_i; \mathrm{NN}^{\mathrm{dec}}_{\theta_{\mu}}(z_i), \theta_{\sigma^2} I_d)
	\quad
	\text{and}
	\quad
	p(z_i) = \mathcal{N}(0, I_d).
\end{align}
$$
Here $\mathrm{NN}$ denotes a neural network and $\theta_{\mu}$ denote the weights. 

Training is done via maximum likelihood using variational Bayes with an amortised  posterior approximation 
$$
\begin{align}
q_{\phi}(z_i;x) = \mathcal{N}(z_i;\mathrm{NN}^{\mathrm{enc-mean}}_{\phi_{\mu}}(x_i), \mathrm{diag}(\mathrm{NN}^{\mathrm{enc-var}}_{\theta_{\sigma}}(x_i))
\end{align}
$$
Please check the lecture slides/video for more details.

The function to optimise is the negative evidence lower bound for this model
$$
\begin{align}
	L(\theta, \phi; X) = 
	-\sum_i E_{q_{\phi}(z_i ; x_i)}[\log p_{\theta}(x_i \vert x_i)] 
	+ \mathrm{KL}[q_{\phi}(z_i; x_i) \vert\!\vert p(z_i)].
\end{align}
$$

which now also has to be optimised w.r.t. the model parameters $\theta$. Note that we did not have this in the logistic regression model.

**Implementation** The parameters of the distributions are implemented via a pair of neural networks. The decoder implements  $p_{\theta}(x_i \vert z_i)$ while the encoder implements $q_{\phi}(z_i; x_i) $ .

```
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
				...
        self.model  = nn.Sequential(...)
    def forward(self, data):
        input  = data
        output = self.model(input)
        loc, scale_isp = torch.split(output, [self.d_state, self.d_state], dim=-1)
        return loc, STD_MIN + torch.nn.functional.softplus(scale_isp)
        
 class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
				...
        self.loc  = nn.Sequential(...)
        self.scale_isp = nn.Parameter(0.5*torch.ones(1), requires_grad=True)

    def forward(self, input:
        loc   = self.loc(input)
        scale = STD_MIN + torch.nn.functional.softplus(self.scale_isp) * torch.ones(loc.shape, device=loc.device) 
        return loc, scale        
```

 

### Key variational inference concepts to learn from this model

The loss function is implemented in 

```
def loss(self, imgs):
```

#### Expectation Maximisation

In this model we not only have to approximate $p_{\theta}(z_i \vert x_i)$ but we also need to maximise the lower bound with respext to the model parameters $\theta$., that is we jointly maximise the w.r.t. the model parameters and the posterior approximation $q_{\phi}(z_i;x_i)$.

**Implementation**  joint learning is implemented via a single optimiser

```
def configure_optimizers(self):
    opt = torch.optim.Adam(itertools.chain(self.encoder.parameters(),
                                           self.decoder.parameters()),
                               lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
    return opt
```

#### Amortised variational inference

If we would proceed according to the logistic regression model we would have to approximate each $p_{\theta}(z_i\vert x_i)$ in a separate inner loop for each new $\theta$ value. Instead we learn $q_{\phi}(z_i; x_i) \approx p_{\theta}(z_i \vert x_i)$ thus replacing the variational inference algorithm to learn $q_{\phi}(z_i; x_i)$ with learning the parameter mappings $\mathrm{NN}^{\mathrm{enc-mean}}_{\phi_{\mu}}(x_i)$ and $\mathrm{NN}^{\mathrm{enc-var}}_{\theta_{\mu}}(x_i)$.  

```
q_zx = torch.distributions.normal.Normal(*self.encoder(imgs.view([size_batch] + self.encoder.shape_input)))
z_samples = q_zx.rsample((self.hparams.n_samples,))
p_xz = torch.distributions.normal.Normal(*self.decoder(z_samples.view([self.hparams.n_samples, size_batch] + self.decoder.shape_input)))
lik_eval = p_xz.log_prob(imgs.view([size_batch] + self.encoder.shape_input))
logp = torch.sum(torch.mean(torch.mean(lik_eval, dim=0, keepdim=True), dim=1, keepdim=True))
```



### Optional questions and tasks

- Run the code with `python run.py --config config_vae.yaml`and check metrics with `tensorboard  --logdir lightning_logs`.

- If you have time, read the original paper https://arxiv.org/pdf/1312.6114.pdf .

- Try to answer the questions
  - how is this model different from the logistic regression one
    - in terms of latent variables?
    - in terms of likelihood model?
    
  - what are the differences in term of approximate inference
    - in term of latent variables?
    - in terms of parameterisation, what does amortisation mean?
    - can we use amortisation for the logistic regression model?
    
    

