import os
from collections import OrderedDict
import itertools
import yaml

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.distributions import Normal, MultivariateNormal ,Bernoulli
from torch.distributions.kl import kl_divergence

import importlib

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

STD_MIN = 1e-8
TORCH_PI = torch.tensor(math.pi)

class LikelihoodLogistic(nn.Module):
    def __init__(
        self,
        n_nodes_gh = 9
    ):
        super().__init__()
    
        self.n_nodes_gh = n_nodes_gh 
        xgh, wgh = np.polynomial.hermite.hermgauss(self.n_nodes_gh) 
        xgh      = np.sqrt(2.0) * xgh
        wgh      = wgh/np.sqrt(np.pi) 
        logwgh   = -0.5*np.log(np.pi) + np.log(wgh)

        self.xgh = nn.Parameter(torch.tensor(xgh, dtype=torch.float32).reshape((1,-1)), requires_grad=False) 
        self.wgh = nn.Parameter(torch.tensor(wgh, dtype=torch.float32).reshape((1,-1)), requires_grad=False) 
        self.logwgh = nn.Parameter(torch.tensor(logwgh, dtype=torch.float32).reshape((1,-1)), requires_grad=False) 
        self.torch_pi = nn.Parameter(torch.tensor(math.pi, dtype=torch.float32), requires_grad=False)

    def logp_expect(self, labels_sign, loc, scale):
        # warning, dim=-1 broadcasting
        n_terms = loc.shape[0]
        z = labels_sign.repeat((1, self.n_nodes_gh))*(loc.repeat((1,self.n_nodes_gh)) + torch.matmul(scale, self.wgh))  
        logp_expct = -torch.sum(torch.log(1+torch.exp(-z)*self.wgh.repeat((n_terms,1))), dim=-1, keepdim=True)

        return logp_expct
    
    def logp_eval(self, z, labels_sign):
        return -torch.log(1+torch.exp(-labels_sign*z))

    def predict(self, loc, scale, quadrature=True):
        n_terms = loc.shape[0]
        if quadrature:
            z = loc.repeat((1,self.n_nodes_gh)) + torch.matmul(scale, self.wgh)  
            prob_expct = torch.sum((1/(1+torch.exp(-z)))*self.wgh.repeat((n_terms,1)), dim=-1, keepdim=True)
        else:
            z = (1/torch.sqrt(1+(self.torch_pi/8.0)*(scale**2)))*loc
            prob_expct = 1/(1+torch.exp(-z))
 
        return prob_expct


class ModelLogisicRegressionMvn(LightningModule):
    def __init__(
        self,
        dim,
        size_data,
        scale_prior = 1.0,
        n_nodes_quadrature= 9,
        n_samples_mc =8,
        optimizer_name="Adam",
        optimizer_lr = 0.001,
        save_path="runs/models/debug"
    ):
        super().__init__()
        
        # hyperparameters
        hparams = {}
        hparams.update({"dim": dim,
                        "size_data": size_data,
                        "n_nodes_quadrature": n_nodes_quadrature,
                        "n_samples_mc": n_samples_mc,
                        "scale_prior": scale_prior,
                        "optimizer_name": optimizer_name,
                        "optimizer_lr": optimizer_lr})
        self.save_hyperparameters(hparams)
        if True:
            print(self.hparams)

        self.dim          = dim
        self.size_data    = size_data
        self.n_nodes_gh   = n_nodes_quadrature
        self.n_samples_mc = n_samples_mc
        self.loc_prior    = torch.tensor(0.0, dtype=torch.float32)
        self.scale_prior  = torch.tensor(scale_prior, dtype=torch.float32)
        self.optimizer_name = optimizer_name
        self.optimizer_lr = optimizer_lr

        # configs
        self.weights_loc           = nn.Parameter(torch.zeros((self.dim,1)), requires_grad=True) 
        self.weights_scale_logdiag = nn.Parameter(torch.zeros((self.dim)), requires_grad=True) 
        self.weights_scale_lower   = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True) 

        # this is also needed for prediction
        self.logit = LikelihoodLogistic(n_nodes_quadrature)
        
        #saving
        self.save_path = save_path

    def configure_callbacks(self):
        checkpoint_validate = ModelCheckpoint(monitor="loss_val",
                                              dirpath=self.save_path,
                                              filename = "loss_val-{epoch}-{step}",
                                              mode="min")
        return [checkpoint_validate]

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), lr=self.optimizer_lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        features, labels = batch
        loss, logp, kl_div = self.loss(features, labels)
        acc = self.accuracy(features, labels)

        self.log_dict({'loss': loss.detach(), "logp": logp.detach(), "kl": kl_div.detach(), "acc": acc.detach()},
                  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        tqdm_dict = {'loss': loss.detach(), "acc": acc.detach()}
        output = OrderedDict({'loss': loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})

        #print(f"accuracy: {acc}")

        return output

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        loss, logp, kl_div = self.loss(features, labels)
        acc = self.accuracy(features, labels)

        self.log_dict({'loss_val': loss.detach(), "logp_val": logp.detach(), "kl_val": kl_div.detach(), "acc": acc.detach()},
                  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        tqdm_dict = {'loss_val': loss.detach(), "acc_val": acc.detach()}
        output = OrderedDict({'loss_val': loss.detach(), 'progress_bar': tqdm_dict, 'log': tqdm_dict})

        return output

    def test_step(self, batch, batch_idx):
        features, labels = batch
        loss, logp, kl_div = self.loss(features, labels)
        acc = self.accuracy(features, labels)

        self.log_dict({'loss_test': loss.detach(), "logp_test": logp.detach(), "kl_test": kl_div.detach(), "acc_test": acc.detach()},
                  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        tqdm_dict = {'loss_test': loss.detach(), "acc_test": acc.detach()}
        output = OrderedDict({'loss_test': loss.detach(), 'progress_bar': tqdm_dict, 'log': tqdm_dict})

        return output

    def weights_chol(self):
        return torch.tril(self.weights_scale_lower, -1) + torch.diag(torch.exp(self.weights_scale_logdiag))

    def forward(self, features):
        L   = self.weights_chol() 
        z_loc     = torch.matmul(features, self.weights_loc)
        z_scale   = torch.sqrt(torch.sum(torch.matmul(features, L)**2, dim=-1, keepdim=True))
        p_labels  = self.logit.predict(z_loc, z_scale, quadrature=True)

        return p_labels

    def accuracy(self, features, labels):
        p_labels = self.forward(features)
        labels_pred = (p_labels > 0.5).float()

        return torch.mean(torch.isclose(labels, labels_pred).float())

    def loss(self, features, labels):
        # computing expected negative log likelihood
        # reparameterisation of stochastic variables
        L       = self.weights_chol() 
        p_post  = MultivariateNormal(loc=self.weights_loc.squeeze(), scale_tril=L)

        # local reparameterisation and MCsampling
        z_loc     = torch.matmul(features, self.weights_loc).squeeze()
        z_scale   = torch.sqrt(torch.sum(torch.matmul(features, L)**2, dim=-1, keepdim=True)).squeeze()
        z_samples = Normal(loc=z_loc, scale=z_scale).rsample([self.n_samples_mc]).transpose(0,1)

        # data distribution via MC samples
        p_labels   = Bernoulli(logits=z_samples)
        # computing the MC samples based expected log likelihood with batch learning correction
        logp_expct = self.size_data*torch.mean(p_labels.log_prob(labels.repeat((1, self.n_samples_mc))))

        # computing KL
        p_prior = MultivariateNormal(loc=torch.zeros_like(self.weights_scale_logdiag),
                                    scale_tril=self.scale_prior*torch.diag(torch.ones_like(self.weights_scale_logdiag)))
        kl_div  = kl_divergence(p_post, p_prior) 

        # compute ELBO
        loss = -logp_expct + kl_div

        return loss, logp_expct, kl_div 

   
