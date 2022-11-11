import os
from collections import OrderedDict
import itertools
import yaml

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

import torchvision

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

STD_MIN = 1e-8

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.shape_input = self.config.shape_input
        self.shape_output = self.config.shape_output

        self.model  = nn.Sequential(OrderedDict(
                        [(name, getattr(nn, fname)(*args, **kwargs)) for name, fname, args, kwargs in config.arch]))

        self.d_state = config.shape_output[-1] // 2

    def forward(self, data):
        #input  = data.view(data.size(0), *self.config.shape_input)
        #output = self.model(input).view(data.size(0),*self.config.shape_output)
        input  = data
        output = self.model(input)
        loc, scale_isp = torch.split(output, [self.d_state, self.d_state], dim=-1)

        return loc, STD_MIN + torch.nn.functional.softplus(scale_isp)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shape_input = self.config.shape_input
        self.shape_output = self.config.shape_output

        self.loc  = nn.Sequential(OrderedDict(
                        [(name, getattr(nn, fname)(*args, **kwargs)) for name, fname, args, kwargs in config.arch]))
        self.scale_isp = nn.Parameter(0.5*torch.ones(1), requires_grad=True)

    def forward(self, state):
        #input  = state.view(state.size(0), *self.config.shape_input)
        #loc   = self.loc(input).view(state.size(0),*self.config.shape_output)
        input = state
        loc   = self.loc(input)
        scale = STD_MIN + torch.nn.functional.softplus(self.scale_isp) * torch.ones(loc.shape, device=loc.device)
        
        return loc, scale 


class ModelVae(LightningModule):

    def __init__(
        self,
        config_model,
        config_optimizers,
        config_saving = None,
        config_to_save = None,
        **kwargs
    ):
        super().__init__()
        
        # hyperparameters
        hparams = {}
        hparams.update(config_model)
        hparams.update(config_optimizers)
        self.save_hyperparameters(hparams)
        
        # configs
        self.config_model      = config_model
        self.config_optimizers = config_optimizers
        self.config_saving     = config_saving
        self.config_to_save    = config_to_save

        # sizes
        self.d_state   = self.config_model.decoder.shape_input[-1]
        self.shape_obs = self.config_model.decoder.shape_output 
        
        self.size_batch_last = 0

        # create model
        self.decoder = Decoder(config_model.decoder)
        self.encoder = Encoder(config_model.encoder)
        self._create_prior()

        # something to display
        self.z_val = self.prior.sample((64,))
        self.shape_data = None

        # admin saving
        self.save_path = self._assemble_save_path()
        self._save_config() 
 
    def _create_prior(self):
        self.register_buffer("prior_loc", torch.zeros(self.d_state))
        self.register_buffer("prior_scale", torch.ones(self.d_state))
        self.prior = torch.distributions.normal.Normal(loc=self.prior_loc, scale=self.prior_scale)

    def _assemble_save_path(self):
        return self.config_saving.save_path


    def _save_config(self):
        if self.config_to_save is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            with open(os.path.join(self.save_path, "config.yaml"), "w") as file:
                yaml.dump(dict(self.config_to_save), file, default_flow_style=False)


    def configure_callbacks(self):

        #checkpoint_train    = ModelCheckpoint(monitor="loss",
        #                                      dirpath=self.save_path,
        #                                      filename = "loss-{epoch}-{step}",
        #                                      mode="min")
        checkpoint_validate = ModelCheckpoint(monitor="loss_val",
                                              dirpath=self.save_path,
                                              filename = "loss_val-{epoch}-{step}",
                                              mode="min")

        
        #return [checkpoint_train, checkpoint_validate]
        return [checkpoint_validate]



    def configure_optimizers(self):
        opt = torch.optim.Adam(itertools.chain(self.encoder.parameters(),
                                               self.decoder.parameters()),
                               lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        return opt


    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        size_batch = imgs.size(0)

        if self.shape_data == None:
            self.shape_data = list(imgs.shape[1:])

        if self.size_batch_last != size_batch:
            self.size_batch_last = size_batch
            self.prior_batch = torch.distributions.normal.Normal(
                            loc=torch.zeros((size_batch, self.d_state), device=self.device),
                            scale=torch.ones((size_batch, self.d_state), device=self.device)
                            )
   
        data = imgs.view([size_batch] + self.encoder.shape_input)

        q_zx      = torch.distributions.normal.Normal(*self.encoder(data))
        z_samples = q_zx.rsample((self.hparams.n_samples,))
        p_xz      = torch.distributions.normal.Normal(*self.decoder(z_samples))
        lik_eval  = p_xz.log_prob(data)

       
        if False:
            with torch.no_grad():
                ll = []
                for i in range(z_samples.size(0)):
                    p  = torch.distributions.normal.Normal(*self.decoder(z_samples[i]))
                    ll.append(p.log_prob(data))
                ll = torch.stack(ll, dim=0)
                delta = torch.max(torch.abs(ll-lik_eval))
            breakpoint()

        logp       = torch.sum(
                        torch.mean(
                            torch.mean(lik_eval, dim=0, keepdim=True), dim=1, keepdim=True))
        divergence = torch.sum(
                        torch.mean(torch.distributions.kl_divergence(q_zx, self.prior_batch),dim=0))

        d = self.trainer.max_epochs // 4
        if self.current_epoch < d:
            beta = torch.tensor(0.0, device=self.device)
        elif  self.current_epoch > 2*d:
            beta = torch.tensor(1.0, device=self.device)
        else:
            beta = torch.tensor(
                    (self.current_epoch -d)/d,
                    device=self.device
                    )

        loss = -logp + beta * divergence

        self.log_dict({'loss': loss, "logp": logp, "div": divergence, "beta": beta, "std": torch.nn.functional.softplus(self.decoder.scale_isp)},
                  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        tqdm_dict = {'loss': loss.detach()}
        output = OrderedDict({'loss': loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})

        return output


    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        size_batch = imgs.size(0)
        if self.shape_data == None:
            self.shape_data = list(imgs.shape[1:])

        if self.size_batch_last != size_batch:
            self.size_batch_last = size_batch
            self.prior_batch = torch.distributions.Normal(
                            loc=torch.zeros((size_batch, self.d_state)).to(self.device),
                            scale=torch.ones((size_batch, self.d_state)).to(self.device)
                            )

        q_zx      = torch.distributions.normal.Normal(*self.encoder(imgs.view([size_batch] + self.encoder.shape_input)))
        z_samples = q_zx.rsample((self.hparams.n_samples,))
        p_xz      = torch.distributions.normal.Normal(*self.decoder(z_samples.view([self.hparams.n_samples, size_batch] + self.decoder.shape_input)))
        lik_eval  = p_xz.log_prob(imgs.view([size_batch] + self.encoder.shape_input))


        logp       = torch.sum(
                        torch.mean(
                            torch.mean(lik_eval, dim=0, keepdim=True), dim=1, keepdim=True))
        divergence = torch.sum(
                        torch.mean(torch.distributions.kl_divergence(q_zx, self.prior_batch),dim=0))

        loss = -logp + divergence

   
        self.log_dict({'loss_val': loss, "logp_val": logp, "div_val": divergence},
                  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        tqdm_dict = {'loss_val': loss.detach()}
        output = OrderedDict({'loss_val': loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})

        return output

    def on_validation_epoch_end(self):
        z_val = self.z_val.type_as(self.decoder.loc[0].weight)
        # log sampled images
        p_xz  = torch.distributions.normal.Normal(*self.decoder(z_val))
        
        if False:
            sample_imgs = p_xz.sample().view([z_val.size(0)]+self.shape_data)
        else:
             sample_imgs =0.5* p_xz.loc.view([z_val.size(0)]+self.shape_data) + 0.5

        grid = torchvision.utils.make_grid(1-sample_imgs, nrow=4)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

