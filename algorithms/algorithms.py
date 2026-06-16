import torch
import torch.nn as nn
import numpy as np
import itertools    
import higher
from models.models import CNN, classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc, CNN_ATTN, SWL_Discriminator, WeightAllocator, ActivityClassifier, FeatureExtracter
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss, NTXentLoss, SupConLoss, SinkhornDistance
from utils import EMA, AverageMeter
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F
import mlflow
from models.HAR_Model import HARmodel
from models.kCMMD_loss import cmmd_loss as kCMMD_loss
from models.Consistencyloss import kl_div_loss as consistency_kl_loss
import math
from models.acon_modules import FrequencyEncoder, FrequencyClassifierHead, TemporalClassifierHead, ACON_Discriminator
from models.raincoat_modules import tf_encoder, tf_decoder, Raincoat_classifier
from models.ssss_tsa_modules import SSSS_SepReps_with_multihead
from models.cluda_modules import CLUDA_Network, CLUDA_Augmenter

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)


    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter,val_loader, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None
        val_loss_meter = AverageMeter()
        
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for key in avg_meter.keys():
                avg_meter[key].reset()
            val_loss_meter.reset()


            # training loop 
            self.network.train() # Set to train mode
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # --- Validation Loop (For Logging Only) ---
            self.network.eval() # Set to eval mode
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                    
                    val_feat = self.feature_extractor(val_x)
                    val_pred = self.classifier(val_feat)
                    
                    val_loss = self.cross_entropy(val_pred, val_y)
                    
                    # Use val_x.size(0) for the batch size
                    val_loss_meter.update(val_loss.item(), val_x.size(0))               
            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')  
            # mlflow.log_metrics(avg_meter,step=epoch)

            metrics_to_log = {key: val.avg for key, val in avg_meter.items()}
            
            mlflow.log_metrics(metrics_to_log, step=epoch)
            # logger.debug(f"MLFLOW_METRICS: {metrics_to_log}")   
            logger.debug(f'-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model, best_model
    
    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError
       

class NO_ADAPT(Algorithm):
    """
    Lower bound: train on source and test on target.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:
            
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()
    

class TARGET_ONLY(Algorithm):
    """
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        for trg_x, trg_y in trg_loader:

            trg_x, trg_y = trg_x.to(self.device), trg_y.to(self.device)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            trg_cls_loss = self.cross_entropy(trg_pred, trg_y)

            loss = trg_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Trg_cls_loss': trg_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, trg_x.size(0))

        self.lr_scheduler.step()


class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # correlation alignment loss
        self.coral = CORAL()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # add if statement

        # if len(src_loader) > len(trg_loader):
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'coral_loss': coral_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()

class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)
            mmd_loss = self.mmd(src_feat, trg_feat)
            cond_ent_loss = self.cond_ent(trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["mmd_wt"] * mmd_loss + \
                self.hparams["cond_ent_wt"] * cond_ent_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                    'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        
        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))
    
        num_batches = min(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
           
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()

class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment Losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            # source features and predictions
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)
            pred_concat = torch.cat((src_pred, trg_pred), dim=0)

            # Domain classification loss
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            # loss of domain discriminator according to fake labels

            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))
        self.lr_scheduler.step()

class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Aligment losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)
        self.ema = EMA(0.998)
        self.ema.register(self.network)

        # Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
       
    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)

            # Domain classification loss
            disc_prediction = self.domain_classifier(feat_concat.detach())
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            disc_prediction = self.domain_classifier(feat_concat)

            # loss of domain discriminator according to fake labels
            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # Virual advariarial training loss
            loss_src_vat = self.vat_loss(src_x, src_pred)
            loss_trg_vat = self.vat_loss(trg_x, trg_pred)
            total_vat = loss_src_vat + loss_trg_vat
            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

            # update exponential moving average
            self.ema(self.network)

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()

class DSAN(Algorithm):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Alignment losses
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)        # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate lmmd loss
            domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()

class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # aligment losses
        self.coral = CORAL()
        self.HoMM_loss = HoMM_loss()

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))


        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate lmmd loss
            domain_loss = self.HoMM_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

            
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate mmd loss
            domain_loss = self.mmd_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()

class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Domain classifier
        self.domain_classifier = Discriminator(configs)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
            
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

        num_batches = min(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
        
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()

class AdvSKM(Algorithm):
    """
    AdvSKM: https://www.ijcai.org/proceedings/2021/0378.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()
        self.AdvSKM_embedder = AdvSKM_Disc(configs).to(device)
        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features
            
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
            target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
            mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss.requires_grad = True

            # update discriminator
            self.optimizer_disc.zero_grad()
            mmd_loss.backward()
            self.optimizer_disc.step()

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # domain loss.
            source_embedding_disc = self.AdvSKM_embedder(src_feat)
            target_embedding_disc = self.AdvSKM_embedder(trg_feat)

            mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss_adv.requires_grad = True

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * mmd_loss_adv + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            # update optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': mmd_loss_adv.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                    avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()

class SASA(Algorithm):
    
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # feature_length for classifier
        configs.features_len = 1
        self.classifier = classifier(configs)
        # feature length for feature extractor
        configs.features_len = 1
        self.feature_extractor = CNN_ATTN(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))
    
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # Extract features
            src_feature = self.feature_extractor(src_x)
            tgt_feature = self.feature_extractor(trg_x)

            # source classification loss
            y_pred = self.classifier(src_feature)
            src_cls_loss = self.cross_entropy(y_pred, src_y)

            # MMD loss
            domain_loss_intra = self.mmd_loss(src_struct=src_feature,
                                            tgt_struct=tgt_feature, weight=self.hparams['domain_loss_wt'])

            # total loss
            total_loss = self.hparams['src_cls_loss_wt'] * src_cls_loss + domain_loss_intra

            # remove old gradients
            self.optimizer.zero_grad()
            # calculate gradients
            total_loss.backward()
            # update the weights
            self.optimizer.step()

            losses =  {'Total_loss': total_loss.item(), 'MMD_loss': domain_loss_intra.item(),
                    'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()
    def mmd_loss(self, src_struct, tgt_struct, weight):
        delta = torch.mean(src_struct - tgt_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value


class CoTMix(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

         # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)
        self.entropy_loss = ConditionalEntropyLoss()
        self.sup_contrastive_loss = SupConLoss(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 

        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # ====== Temporal Mixup =====================
            src_dominant, trg_dominant = self.temporal_mixup(src_x, trg_x)

            # ====== Source =====================
            self.optimizer.zero_grad()

            # Src original features
            src_orig_feat = self.feature_extractor(src_x)
            src_orig_logits = self.classifier(src_orig_feat)

            # Target original features
            trg_orig_feat = self.feature_extractor(trg_x)
            trg_orig_logits = self.classifier(trg_orig_feat)

            # -----------  The two main losses
            # Cross-Entropy loss
            src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
            loss = src_cls_loss * round(self.hparams["src_cls_weight"], 2)

            # Target Entropy loss
            trg_entropy_loss = self.entropy_loss(trg_orig_logits)
            loss += trg_entropy_loss * round(self.hparams["trg_entropy_weight"], 2)

            # -----------  Auxiliary losses
            # Extract source-dominant mixup features.
            src_dominant_feat = self.feature_extractor(src_dominant)
            src_dominant_logits = self.classifier(src_dominant_feat)

            # supervised contrastive loss on source domain side
            src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
            src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
            loss += src_supcon_loss * round(self.hparams["src_supCon_weight"], 2)

            # Extract target-dominant mixup features.
            trg_dominant_feat = self.feature_extractor(trg_dominant)
            trg_dominant_logits = self.classifier(trg_dominant_feat)

            # Unsupervised contrastive loss on target domain side
            trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
            loss += trg_con_loss * round(self.hparams["trg_cont_weight"], 2)

            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(),
                    'src_cls_loss': src_cls_loss.item(),
                    'trg_entropy_loss': trg_entropy_loss.item(),
                    'src_supcon_loss': src_supcon_loss.item(),
                    'trg_con_loss': trg_con_loss.item()
                    }
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()           

    def temporal_mixup(self,src_x, trg_x):
        
        mix_ratio = round(self.hparams["mix_ratio"], 2)
        temporal_shift = self.hparams["temporal_shift"]
        h = temporal_shift // 2  # half

        src_dominant = mix_ratio * src_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(trg_x, -i, 2) for i in range(-h, h)], 2), 2)

        trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(src_x, -i, 2) for i in range(-h, h)], 2), 2)
        
        return src_dominant, trg_dominant
    


# Untied Approaches: (MCD)
class MCD(Algorithm):
    """
    Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
    MCD: https://arxiv.org/pdf/1712.02560.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.classifier2 = classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)


        # optimizer and scheduler
        self.optimizer_fe = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
                # optimizer and scheduler
        self.optimizer_c1 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
                # optimizer and scheduler
        self.optimizer_c2 = torch.optim.Adam(
            self.classifier2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c1 = StepLR(self.optimizer_c1, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c2 = StepLR(self.optimizer_c2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            
            # source pretraining loop 
            self.pretrain_epoch(src_loader, avg_meter)

            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

            metrics_to_log = {key: val.avg for key, val in avg_meter.items()}
            
            mlflow.log_metrics(metrics_to_log, step=epoch)

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader,avg_meter):
        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
          
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier(src_feat)
            src_pred2 = self.classifier2(src_feat)

            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)

            loss = src_cls_loss1 + src_cls_loss2

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            loss.backward()

            self.optimizer_c1.step()
            self.optimizer_c2.step()
            self.optimizer_fe.step()

            
            losses = {'Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))
        joint_loader =enumerate(zip(src_loader, (trg_loader)))


        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            

            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier(src_feat)
            src_pred2 = self.classifier2(src_feat)

            # source losses
            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)
            loss_s = src_cls_loss1 + src_cls_loss2
            

            # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = False
            # update C1 and C2 to maximize their difference on target sample
            trg_feat = self.feature_extractor(trg_x) 
            trg_pred1 = self.classifier(trg_feat.detach())
            trg_pred2 = self.classifier2(trg_feat.detach())


            loss_dis = self.discrepancy(trg_pred1, trg_pred2)

            loss = loss_s - loss_dis
            
            loss.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            # Freeze the classifiers
            for k, v in self.classifier.named_parameters():
                v.requires_grad = False
            for k, v in self.classifier2.named_parameters():
                v.requires_grad = False
                        # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = True
            # update feature extractor to minimize the discrepaqncy on target samples
            trg_feat = self.feature_extractor(trg_x)        
            trg_pred1 = self.classifier(trg_feat)
            trg_pred2 = self.classifier2(trg_feat)


            loss_dis_t = self.discrepancy(trg_pred1, trg_pred2)
            domain_loss = self.hparams["domain_loss_wt"] * loss_dis_t 

            domain_loss.backward()
            self.optimizer_fe.step()

            self.optimizer_fe.zero_grad()
            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()


            losses =  {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler_fe.step()
        self.lr_scheduler_c1.step()
        self.lr_scheduler_c2.step()

    def discrepancy(self, out1, out2):

        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))



class SWL_Adapt(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # Additional networks required by SWL_Adapt
        self.domain_classifier = SWL_Discriminator(configs)
        self.weight_allocator = WeightAllocator(hparams['WA_N_hid'])
        
        # Move all sub-modules to the correct device
        self.to(device) 
        
        # Optimizers for each component
        self.optimizer_fe = torch.optim.Adam(self.feature_extractor.parameters(), lr=hparams["learning_rate"])
        self.optimizer_d = torch.optim.Adam(self.domain_classifier.parameters(), lr=hparams["learning_rate"])
        self.optimizer_ac = torch.optim.Adam(self.classifier.parameters(), lr=hparams["learning_rate"])
        self.optimizer_wa = torch.optim.Adam(self.weight_allocator.parameters(), lr=hparams["WA_lr"])

        # Schedulers
        self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_d = StepLR(self.optimizer_d, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_ac = StepLR(self.optimizer_ac, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_wa = StepLR(self.optimizer_wa, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        
        self.hparams = hparams
        self.device = device

    def ld_weight(self, logits_d, logits_ac, y=None, yd=None):
        batch_size = logits_d.shape[0]
        with torch.no_grad():
            criterion_d = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
            loss_d = criterion_d(logits_d, yd)
            criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
            if y is None:
                y = logits_ac.max(1)[1]
            loss_c = criterion(logits_ac, y)
        
        d_w = self.weight_allocator(align_G=loss_d, clsf=loss_c)

        scale = d_w.sum(dim=0)
        if scale == 0:
            scale = scale + 0.05
        
        d_w = d_w * batch_size / scale.repeat(batch_size, 1)
        return d_w.reshape(-1)

    def get_oll(self, logits_ac_T):
        pseudo_y_T = logits_ac_T.max(1)[1]
        certainty_y_T = logits_ac_T.softmax(dim=1).max(1)[0]
        mask_T = certainty_y_T > self.hparams['confidence_rate']
        loss_c = torch.sum(F.cross_entropy(logits_ac_T, pseudo_y_T, reduction='none') * mask_T.float().detach())
        return loss_c

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Loss functions
        criterion_c = nn.CrossEntropyLoss().to(self.device)
        criterion_ld = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
        joint_loader =enumerate(zip(src_loader, (trg_loader)))

        # # Joint data loader
        # if len(src_loader) > len(trg_loader):
        #     # Source is longer, so cycle the target
        #     joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        # else:
        #     # Target is longer (or same size), so cycle the source
        #     joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))

            
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            
            # Domain labels (source=0, target=1) for BCEWithLogitsLoss
            yd_S = torch.zeros(src_x.size(0), device=self.device)
            yd_T = torch.ones(src_x.size(0), device=self.device)

            # --- Step 1: Update FE and AC on classification loss ---
            self.optimizer_fe.zero_grad()
            self.optimizer_ac.zero_grad()
            
            src_feat = self.feature_extractor(src_x)
            logits_ac_S = self.classifier(src_feat)
            loss_c_S = criterion_c(logits_ac_S, src_y)
            
            trg_feat = self.feature_extractor(trg_x)
            logits_ac_T = self.classifier(trg_feat)
            
            with torch.no_grad():
                pseudo_y_T = logits_ac_T.max(1)[1]
                certainty_y_T = logits_ac_T.softmax(dim=1).max(1)[0]
                mask_C = (certainty_y_T > self.hparams['confidence_rate']).float()
            
            loss_c_T = 0
            if mask_C.sum() > 0:
                loss_c_T = torch.sum(F.cross_entropy(logits_ac_T, pseudo_y_T, reduction='none') * mask_C) / mask_C.sum()
            
            loss_c = loss_c_S + self.hparams['w_c_T'] * loss_c_T
            loss_c.backward()
            self.optimizer_fe.step()
            self.optimizer_ac.step()
            
            avg_meter['Step1_Cls_Loss'].update(loss_c.item(), src_x.size(0))

            # --- Step 2: Update WA with meta-learning ---
            self.optimizer_wa.zero_grad()

            with higher.innerloop_ctx(self.feature_extractor, self.optimizer_fe) as (fmodel, diffopt):
                # Calculate weighted domain loss on the virtual model
                s_feat_virt = fmodel(src_x)
                rev_s_feat_virt = ReverseLayerF.apply(s_feat_virt, 1)
                logits_d_S_virt = self.domain_classifier(rev_s_feat_virt)
                logits_ac_S_virt = self.classifier(s_feat_virt)
                d_w_S = self.ld_weight(logits_d_S_virt, logits_ac_S_virt, y=src_y, yd=yd_S)
                loss_d_S_virt = (criterion_ld(logits_d_S_virt, yd_S) * d_w_S).mean()

                t_feat_virt = fmodel(trg_x)
                rev_t_feat_virt = ReverseLayerF.apply(t_feat_virt, 1)
                logits_d_T_virt = self.domain_classifier(rev_t_feat_virt)
                logits_ac_T_virt = self.classifier(t_feat_virt)
                d_w_T = self.ld_weight(logits_d_T_virt, logits_ac_T_virt, yd=yd_T)
                loss_d_T_virt = (criterion_ld(logits_d_T_virt, yd_T) * d_w_T).mean()
                
                loss_d_virt = loss_d_S_virt + loss_d_T_virt
                diffopt.step(loss_d_virt) # Update virtual model

                # Calculate meta-loss on the updated virtual model
                meta_trg_feat = fmodel(trg_x)
                meta_logits_ac_T = self.classifier(meta_trg_feat)
                meta_loss_c = self.get_oll(meta_logits_ac_T)

                # Update the real Weight Allocator
                meta_loss_c.backward()
                self.optimizer_wa.step()

            avg_meter['Step2_Meta_Loss'].update(meta_loss_c.item(), trg_x.size(0))

            # --- Step 3: Update FE and D with weighted domain loss ---
            self.optimizer_fe.zero_grad()
            self.optimizer_d.zero_grad()
            
            src_feat = self.feature_extractor(src_x)
            rev_s_feat = ReverseLayerF.apply(src_feat, 1)
            logits_d_S = self.domain_classifier(rev_s_feat)
            logits_ac_S = self.classifier(src_feat)
            d_w_S = self.ld_weight(logits_d_S, logits_ac_S, y=src_y, yd=yd_S)
            loss_d_S = (criterion_ld(logits_d_S, yd_S) * d_w_S).mean()

            trg_feat = self.feature_extractor(trg_x)
            rev_t_feat = ReverseLayerF.apply(trg_feat, 1)
            logits_d_T = self.domain_classifier(rev_t_feat)
            logits_ac_T = self.classifier(trg_feat)
            d_w_T = self.ld_weight(logits_d_T, logits_ac_T, yd=yd_T)
            loss_d_T = (criterion_ld(logits_d_T, yd_T) * d_w_T).mean()

            loss_d = loss_d_S + loss_d_T
            loss_d.backward()
            self.optimizer_fe.step()
            self.optimizer_d.step()
            
            avg_meter['Step3_Domain_Loss'].update(loss_d.item(), src_x.size(0))

        # Update learning rate schedulers at the end of the epoch
        self.lr_scheduler_fe.step()
        self.lr_scheduler_d.step()
        self.lr_scheduler_ac.step()
        self.lr_scheduler_wa.step()



class uDAR(Algorithm):
    """
    Implementation of the uDAR pipeline from the user's notebook.
    This algorithm combines:
    1. Temporal Ensembling for target domain pseudo-labels.
    2. Consistency Regularization (KL divergence) on augmented data.
    3. Class-wise MMD (CMMD) loss for domain alignment.

    ---
    **CRITICAL ASSUMPTIONS & MODIFICATIONS:**
    ---
    """
    def __init__(self, backbone, configs, hparams, device):
        # --- Override base __init__ ---
        # We call nn.Module's init, not Algorithm's, because we are
        # replacing the network structure defined in the base Algorithm.
        super(Algorithm, self).__init__() 
        
        self.configs = configs
        self.hparams = hparams
        self.device = device
        
        class JoinedNetwork(nn.Module):
            def __init__(self, configs):
                super().__init__()
                self.feature_extractor = backbone(configs) # Uses your robust CNN
                self.classifier = classifier(configs) # Uses your classifier
                
            def forward(self, x):
                # 1. Get embeddings (features)
                embeddings = self.feature_extractor(x)
                # 2. Get predictions (logits)
                predictions = self.classifier(embeddings)
                # 3. Return BOTH, as uDAR expects
                return predictions, embeddings

        try:
            # We initialize the joined network with your configs
            self.network = JoinedNetwork(configs).to(device)
        except Exception as e:
            raise ImportError(f"Failed to initialize CNN/Classifier. Error: {e}")
        
        # --- Losses ---
        self.cross_entropy = nn.CrossEntropyLoss()
        try:
            self.cmmd_loss = kCMMD_loss
            self.kl_loss = consistency_kl_loss
        except NameError:
             raise ImportError("Failed to load uDAR losses. Check imports at the top of algorithms.py.")

        # --- Optimizer and Scheduler ---
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        
        # --- Temporal Ensembling State ---
        # These will be initialized in the `update` method once we know
        # the target dataset size.
        self.ensemble_predictions = None
        self.ensemble_counts = None

    def _init_ensembling_state(self, trg_loader):
        """Initializes temporal ensembling tensors based on target dataset size."""
        try:
            dataset_len = len(trg_loader.dataset)
        except Exception as e:
            print(f"Error: Could not get length of trg_loader.dataset. {e}")
            print("Please ensure trg_loader is a DataLoader wrapping a Dataset.")
            raise
            
        self.ensemble_predictions = torch.zeros(dataset_len, self.configs.num_classes).to(self.device)
        self.ensemble_counts = torch.zeros(dataset_len).to(self.device)
        print(f"Initialized temporal ensembling state for {dataset_len} target samples.")

    # --- Augmentation Methods (copied from user notebook) ---
    def _DA_Jitter(self, X):
        # Get sigma from hparams, default to 0.05
        sigma = self.hparams.get('sigma', 0.05) 
        myNoise = torch.randn_like(X) * sigma
        return X + myNoise



    def _rotate_features(self, X, features_to_rotate=12):
        # Get degree (r_theta) from hparams, default to 10
        degree = self.hparams.get('r_theta', 10) 
        radians = torch.tensor(degree * (math.pi / 180)).to(self.device)        # Ensure rotation matrix is on the correct device and dtype
        # Ensure rotation matrix is on the correct device and dtype
        rotation_matrix = torch.tensor([
            [torch.cos(radians), -torch.sin(radians)],
            [torch.sin(radians), torch.cos(radians)]
        ], device=self.device, dtype=X.dtype) 

        # This check from your notebook means rotation is a no-op for 3D data
        if X.ndim == 2 and X.shape[1] >= features_to_rotate:
            rotated = torch.matmul(X[:, :features_to_rotate].reshape(-1, 2), rotation_matrix)
            return torch.cat((rotated.reshape(X.shape[0], -1), X[:, features_to_rotate:]), dim=1)
        else:
            return X

    def _augment_data(self, X):
        """Applies jitter and (if data is 2D) rotation augmentations."""
        jittered_data = self._DA_Jitter(X)
        augmented_data = self._rotate_features(jittered_data)
        return augmented_data

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
            """
            Custom update method for uDAR.
            Initializes ensembling state and runs the epoch loop.
            
            *** NOTE: This version overrides the base 'update' method
            to remove the validation loop, which was causing a crash
            because uDAR does not have a 'self.feature_extractor'. ***
            """
            
            # --- Initialize Ensembling State (one-time) ---
            if self.ensemble_predictions is None:
                self._init_ensembling_state(trg_loader)
            
            # --- Copied from base Algorithm.update ---
            best_src_risk = float('inf')
            best_model = None
            
            # We create a simple val_loss_meter here, though it won't be used
            # (This just prevents other potential errors if it's referenced)
            val_loss_meter = AverageMeter() 
            
            for epoch in range(1, self.hparams["num_epochs"] + 1):

                for key in avg_meter.keys():
                    avg_meter[key].reset()
                val_loss_meter.reset()


                # --- 1. Training loop ---
                self.network.train() # Set to train mode
                self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

                
                # --- 2. Validation Loop (REMOVED) ---
                # The original validation loop was here.
                # We remove it because self.feature_extractor doesn't exist.
                # ---
                
                
                # --- 3. Saving the best model (based on training loss) ---
                if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                    best_src_risk = avg_meter['Src_cls_loss'].avg
                    best_model = deepcopy(self.network.state_dict())


                # --- 4. Logging ---
                logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                for key, val in avg_meter.items():
                    logger.debug(f'{key}\t: {val.avg:2.4f}')   
                
                # Log to MLflow
                metrics_to_log = {key: val.avg for key, val in avg_meter.items()}
                # We can log the dummy validation loss (which will be 0)
                # metrics_to_log['Val_loss'] = val_loss_meter.avg 
                mlflow.log_metrics(metrics_to_log, step=epoch)
                
                logger.debug(f'-------------------------------------')
            
            last_model = self.network.state_dict()

            return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
            
            # --- Get Hyperparameters ---
            weight_cmmd = self.hparams.get("cmmd_weight", 1.0)
            weight_kl = self.hparams.get("kl_weight", 1.0)
            alpha = self.hparams.get("temporal_alpha", 0.6)
            num_classes = self.configs.num_classes
            joint_loader =enumerate(zip(src_loader, (trg_loader)))

            # --- Data Iteration (Using framework's standard) ---
            # if len(src_loader) > len(trg_loader):
            #     joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
            #     n_batches = len(src_loader)
            # else:
            #     joint_loader = enumerate(zip(itertools.cycle(src_loader), trg_loader))
            #     n_batches = len(trg_loader)
                
            # if n_batches == 0:
            #     print("Warning: One of the data loaders has length 0. Skipping epoch.")
            #     return

            # --- THIS IS THE FIX ---
            # Unpack all three items from the target loader: (x, y, index)
            # We don't use trg_y, but we must unpack it.
            for step, ((src_x, src_y), (trg_x, trg_y, trg_indices)) in joint_loader:

                src_x, src_y = src_x.to(self.device), src_y.to(self.device)
                # Assign the correct variables to the device
                trg_x, trg_indices = trg_x.to(self.device), trg_indices.to(self.device)

                # --- Augmentation ---
                src_x_augmented = self._augment_data(src_x)
                trg_x_augmented = self._augment_data(trg_x)
                
                self.optimizer.zero_grad()

                # --- Forward Passes (Original) ---
                src_pred, src_embeddings = self.network(src_x)
                trg_pred, trg_embeddings = self.network(trg_x)

                # --- Forward Passes (Augmented) ---
                src_pred_augmented, _ = self.network(src_x_augmented)
                trg_pred_augmented, _ = self.network(trg_x_augmented)

                # --- 1. Source Classification Loss ---
                loss_source = self.cross_entropy(src_pred, src_y)

                # --- 2. Consistency Loss (KL) ---
                src_log_probs_aug = F.log_softmax(src_pred_augmented, dim=1)
                trg_log_probs_aug = F.log_softmax(trg_pred_augmented, dim=1)
                src_probs = F.softmax(src_pred.detach(), dim=1)
                trg_probs = F.softmax(trg_pred.detach(), dim=1)
                
                loss_source_aug = self.kl_loss(src_log_probs_aug, src_probs)
                loss_target_aug = self.kl_loss(trg_log_probs_aug, trg_probs)
                loss_consistency = loss_source_aug + loss_target_aug

                # --- 3. Temporal Ensembling & Pseudo-labels ---
                trg_pred_softmax = F.softmax(trg_pred, dim=1)
                
                self.ensemble_predictions[trg_indices] *= alpha
                self.ensemble_predictions[trg_indices] += (1 - alpha) * trg_pred_softmax.detach()
                self.ensemble_counts[trg_indices] += 1

                ensemble_avg = self.ensemble_predictions[trg_indices] / self.ensemble_counts[trg_indices].unsqueeze(1)
                trg_pseudo_labels = torch.argmax(ensemble_avg, dim=1)
                
                # --- 4. CMMD Loss ---
                src_emb_flat = src_embeddings.view(src_embeddings.size(0), -1)
                trg_emb_flat = trg_embeddings.view(trg_embeddings.size(0), -1)
                
                loss_cmmd = self.cmmd_loss(
                    src_emb_flat, 
                    src_y, 
                    trg_emb_flat, 
                    trg_pseudo_labels, 
                    num_classes,
                    gamma=self.hparams.get("cmmd_gamma", 1.0),
                    lambda_reg=self.hparams.get("cmmd_lambda_reg", 1e-3)
                )

                # --- Total Loss ---
                loss = loss_source + \
                    (weight_cmmd * loss_cmmd) + \
                    (weight_kl * loss_consistency)

                loss.backward()
                self.optimizer.step()

                # --- Update Logs ---
                losses = {
                    'Total_loss': loss.item(),
                    'Src_cls_loss': loss_source.item(), 
                    'CMMD_loss': loss_cmmd.item(),
                    'Consistency_loss': loss_consistency.item()
                }
                
                for key, val in losses.items():
                    avg_meter[key].update(val, src_x.size(0))

            # --- End of Epoch ---
            self.lr_scheduler.step()


class DAAN(Algorithm):
    """
    DAAN: Dynamic Adversarial Adaptation Networks
    https://arxiv.org/abs/1911.08939
    Combines global and local (per-class) domain alignment with a dynamic
    weighting mechanism (MU) that automatically balances their contributions.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.hparams = hparams
        self.device = device
        self.num_classes = configs.num_classes

        feat_dim = configs.features_len * configs.final_out_channels

        # Global domain discriminator (same layer structure as original DAAN)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 2)
        )

        # Local (per-class) domain discriminators — one per class, same architecture
        self.dcis = nn.ModuleList()
        for i in range(self.num_classes):
            self.dcis.append(nn.Sequential(
                nn.Linear(feat_dim, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 2)
            ))

        # Single optimizer for all parameters (backbone + classifier + discriminators)
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) +
            list(self.domain_classifier.parameters()) +
            list(self.dcis.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # Dynamic MU state (persists across epochs)
        self.D_M = 0
        self.D_C = 0
        self.MU = 0

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Ensure discriminators (which have Dropout) are in train mode.
        # The base update() only calls self.network.train() which covers
        # feature_extractor + classifier but not our extra modules.
        self.domain_classifier.train()
        self.dcis.train()

        joint_loader = enumerate(zip(src_loader, trg_loader))
        num_batches = min(len(src_loader), len(trg_loader))

        # ---- Update MU per epoch (dynamic global/local balance) ----
        if self.D_M == 0 and self.D_C == 0 and self.MU == 0:
            self.MU = 0.5
        else:
            d_m_avg = self.D_M / num_batches
            d_c_avg = self.D_C / num_batches
            self.MU = 1 - d_m_avg / (d_m_avg + d_c_avg + 1e-12)

        d_m = 0
        d_c = 0

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()

            # ---- Extract features ----
            src_feat = self.feature_extractor(src_x)
            trg_feat = self.feature_extractor(trg_x)

            # ---- Classification ----
            src_pred = self.classifier(src_feat)
            trg_pred = self.classifier(trg_feat)
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # Softmax probabilities (used for per-class weighting)
            p_source = F.softmax(src_pred, dim=1)
            p_target = F.softmax(trg_pred, dim=1)

            # ---- Gradient reversal ----
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)

            # ---- Global domain discrimination ----
            s_domain_output = self.domain_classifier(src_feat_reversed)
            t_domain_output = self.domain_classifier(trg_feat_reversed)

            sdomain_label = torch.zeros(len(src_x)).long().to(self.device)
            tdomain_label = torch.ones(len(trg_x)).long().to(self.device)

            err_s_domain = self.cross_entropy(s_domain_output, sdomain_label)
            err_t_domain = self.cross_entropy(t_domain_output, tdomain_label)

            # ---- Local (per-class) domain discrimination ----
            loss_s = 0.0
            loss_t = 0.0
            tmpd_c = 0
            for i in range(self.num_classes):
                # Weight features by class probability
                ps = p_source[:, i].reshape((src_x.shape[0], 1))
                fs = ps * src_feat_reversed
                pt = p_target[:, i].reshape((trg_x.shape[0], 1))
                ft = pt * trg_feat_reversed

                loss_si = self.cross_entropy(self.dcis[i](fs), sdomain_label)
                loss_ti = self.cross_entropy(self.dcis[i](ft), tdomain_label)
                loss_s += loss_si
                loss_t += loss_ti
                tmpd_c += 2 * (1 - 2 * (loss_si + loss_ti))

            tmpd_c /= self.num_classes
            d_c += tmpd_c.cpu().item()

            # ---- Combine losses ----
            global_loss = self.hparams["global_loss_wt"] * (err_s_domain + err_t_domain)
            local_loss = self.hparams["local_loss_wt"] * (loss_s + loss_t)

            d_m += 2 * (1 - 2 * global_loss.cpu().item())

            join_loss = (1 - self.MU) * global_loss + self.MU * local_loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + join_loss

            loss.backward()
            self.optimizer.step()

            losses = {
                'Total_loss': loss.item(),
                'Src_cls_loss': src_cls_loss.item(),
                'Global_loss': global_loss.item(),
                'Local_loss': local_loss.item(),
            }

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        # Persist D_M, D_C for next epoch's MU computation
        self.D_M = d_m
        self.D_C = d_c
        self.lr_scheduler.step()


##################################################
##########  ACON Algorithm  ######################
##################################################

class ACON(Algorithm):
    """
    ACON: Adversarial Consistency for Cross-Domain Time Series Adaptation
    https://openreview.net/pdf?id=cIBSsXowMr

    Uses dual temporal + frequency branches with adversarial domain alignment
    and cross-branch KL consistency.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.hparams = hparams
        self.device = device
        self.num_classes = configs.num_classes

        # --- ACON-specific config params (from data_model_configs) ---
        self.period = configs.period
        self.avg_mode = configs.avg_mode
        self.fft_mode = self.period // 2 + 1
        fft_normalize = getattr(configs, 'fft_normalize', False)
        assert self.avg_mode < self.fft_mode, \
            f"avg_mode ({self.avg_mode}) must be < fft_mode ({self.fft_mode})"

        # --- Temporal branch ---
        # self.feature_extractor is the AdaTime backbone CNN (set by base Algorithm)
        # Its output dim = configs.features_len * configs.final_out_channels
        t_feat_dim = configs.features_len * configs.final_out_channels
        self.classifier = TemporalClassifierHead(t_feat_dim, configs.num_classes)

        # --- Frequency branch ---
        self.f_feature_extractor = FrequencyEncoder(
            configs.input_channels, configs.input_channels, self.fft_mode, fft_normalize
        )
        self.f_classifier = FrequencyClassifierHead(
            self.fft_mode * configs.input_channels, configs.num_classes
        )

        # --- Domain discriminator (operates on outer-product of freq & temporal feats) ---
        self.avg_pooling = nn.AdaptiveAvgPool1d(self.avg_mode)
        disc_in_dim = t_feat_dim * self.avg_mode
        self.domain_classifier = ACON_Discriminator(
            disc_in_dim, hparams['acon_disc_hid_dim']
        )

        # --- Losses ---
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.kl = nn.KLDivLoss(reduction=hparams.get('kl_reduction', 'mean'))
        self.kl_t = hparams.get('kl_t', 1.0)

        # --- Optimizers ---
        # Main optimizer: backbone + temporal classifier + frequency branch
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.classifier.parameters()) +
            list(self.f_feature_extractor.parameters()) +
            list(self.f_classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # Discriminator optimizer (trained separately with true labels)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

    # --- ACON helpers ---

    def period_data(self, x, period):
        """Reshape input into periodic segments for FFT."""
        B, N = x.size(0), x.size(1)
        if x.size(2) % period != 0:
            length = ((x.size(-1) // period) + 1) * period
            padding = torch.zeros([B, N, length - x.size(2)]).to(x.device)
            out = torch.cat([x, padding], dim=2)
        else:
            length = x.size(2)
            out = x
        out = out.reshape(B, N, length // period, period).contiguous()
        return out

    def get_amplitude(self, x_fft):
        """Extract amplitude features from FFT output."""
        a = x_fft.abs()
        if a.dim() == 4:
            a = a.mean(dim=2)
        a_disc = a[:, :, :self.fft_mode]
        a_disc = self.avg_pooling(a_disc.mean(dim=1)).softmax(-1)
        a_cls = a[:, :, :self.fft_mode]
        a_cls = a_cls.reshape(a_cls.size(0), -1)
        return a_cls, a_disc

    # --- Main training logic ---

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Ensure all ACON-specific modules are in train mode
        self.classifier.train()
        self.f_feature_extractor.train()
        self.f_classifier.train()
        self.domain_classifier.train()

        joint_loader = enumerate(zip(src_loader, trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            bs = src_x.size(0)

            # === Temporal branch ===
            src_t_feat = self.feature_extractor(src_x)
            src_t_pred = self.classifier(src_t_feat)

            trg_t_feat = self.feature_extractor(trg_x)
            trg_t_pred = self.classifier(trg_t_feat)

            feat_concat = torch.cat((src_t_feat, trg_t_feat), dim=0)

            # === Frequency branch ===
            src_f_feat = self.f_feature_extractor(self.period_data(src_x, self.period))
            trg_f_feat = self.f_feature_extractor(self.period_data(trg_x, self.period))

            src_a_cls, _ = self.get_amplitude(src_f_feat)
            trg_a_cls, _ = self.get_amplitude(trg_f_feat)

            src_f_pred, src_f_feat_linear = self.f_classifier(src_a_cls, True)
            trg_f_pred, trg_f_feat_linear = self.f_classifier(trg_a_cls, True)

            src_a_disc = self.avg_pooling(src_f_feat_linear).softmax(-1)
            trg_a_disc = self.avg_pooling(trg_f_feat_linear).softmax(-1)

            ft_a_concat = torch.cat([src_a_disc, trg_a_disc], dim=0)

            # === Step 1: Update discriminator with TRUE domain labels ===
            domain_label_src = torch.ones(bs).to(self.device)
            domain_label_trg = torch.zeros(bs).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            feat_x_pred = torch.bmm(
                ft_a_concat.unsqueeze(2), feat_concat.unsqueeze(1)
            ).view(bs * 2, -1).detach()  # detach so disc update doesn't flow into backbone

            disc_prediction = self.domain_classifier(feat_x_pred)
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # === Step 2: Update backbone/classifiers with FAKE domain labels ===
            domain_label_src_fake = torch.zeros(bs).long().to(self.device)
            domain_label_trg_fake = torch.ones(bs).long().to(self.device)
            domain_label_concat_fake = torch.cat((domain_label_src_fake, domain_label_trg_fake), 0)

            feat_x_pred = torch.bmm(
                ft_a_concat.unsqueeze(2), feat_concat.unsqueeze(1)
            ).view(bs * 2, -1)  # no detach — gradients flow back into backbone

            disc_prediction = self.domain_classifier(feat_x_pred)
            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat_fake)

            # Task classification losses (both branches)
            src_t_cls_loss = self.cross_entropy(src_t_pred.squeeze(), src_y)
            src_f_cls_loss = self.cross_entropy(src_f_pred.squeeze(), src_y)

            # Cross-branch KL alignment
            align_s_tf_loss = self.kl(
                F.log_softmax(src_t_pred / self.kl_t, dim=-1),
                F.softmax(src_f_pred / self.kl_t, dim=-1) + 1e-5
            )
            align_t_tf_loss = self.kl(
                F.log_softmax(trg_f_pred / self.kl_t, dim=-1),
                F.softmax(trg_t_pred / self.kl_t, dim=-1)
            )

            # Conditional entropy on target
            entropy_trg_t = self.criterion_cond(trg_t_pred)
            entropy_trg_f = self.criterion_cond(trg_f_pred)

            # Total loss
            loss = self.hparams['cls_trade_off'] * (src_t_cls_loss + src_f_cls_loss) \
                 + self.hparams['domain_trade_off'] * domain_loss \
                 + self.hparams['entropy_trade_off'] * (entropy_trg_t + entropy_trg_f) \
                 + self.hparams['align_t_trade_off'] * align_t_tf_loss \
                 + self.hparams['align_s_trade_off'] * align_s_tf_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {
                'Total_loss': loss.item(),
                'Src_cls_loss': (src_t_cls_loss.item() + src_f_cls_loss.item()),
                'Domain_loss': domain_loss.item(),
                'Align_s_tf_loss': align_s_tf_loss.item(),
                'Align_t_tf_loss': align_t_tf_loss.item(),
                'Entropy_loss': (entropy_trg_t.item() + entropy_trg_f.item()),
            }

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()


##################################################
##########  RAINCOAT Algorithm  ##################
##################################################

class RAINCOAT(Algorithm):
    """
    RAINCOAT: Domain Adaptation for Time Series
    with Reconstruction and Alignment in a Joint Latent Space.
    https://arxiv.org/abs/2302.03133

    Uses a time-frequency encoder/decoder with Sinkhorn OT alignment.
    Has two training phases:
      Phase 1 (update): classification + Sinkhorn alignment + reconstruction
      Phase 2 (correct): reconstruction-only fine-tuning
    """

    def __init__(self, backbone, configs, hparams, device):
        # We call Algorithm.__init__ but override its backbone/classifier.
        # RAINCOAT has its own tf_encoder and classifier.
        super().__init__(configs, backbone)

        self.hparams = hparams
        self.device = device

        # Override the backbone feature_extractor and classifier with RAINCOAT's own
        self.feature_extractor = tf_encoder(configs).to(device)
        self.decoder = tf_decoder(configs).to(device)
        self.classifier = Raincoat_classifier(configs).to(device)
        # Rebuild self.network so base class checkpointing works
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # Main optimizer (encoder + decoder + classifier)
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.decoder.parameters()) +
            list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # Correction optimizer (encoder + decoder only, no classifier)
        self.coptimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.decoder.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # Losses
        self.recons = nn.L1Loss(reduction='sum').to(device)
        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        """Phase 1: Main training with classification + alignment + reconstruction."""
        self.feature_extractor.train()
        self.decoder.train()
        self.classifier.train()

        joint_loader = enumerate(zip(src_loader, trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            self.optimizer.zero_grad()

            # Encode
            src_feat, out_s = self.feature_extractor(src_x)
            trg_feat, out_t = self.feature_extractor(trg_x)

            # Decode (reconstruct)
            src_recon = self.decoder(src_feat, out_s)
            trg_recon = self.decoder(trg_feat, out_t)

            # Reconstruction loss
            recons_loss = 1e-4 * (self.recons(src_recon, src_x) + self.recons(trg_recon, trg_x))

            # Sinkhorn alignment loss
            dr, _, _ = self.sink(src_feat, trg_feat)
            sink_loss = dr

            # Classification loss
            src_pred = self.classifier(src_feat)
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # Combined backward — single pass instead of three retain_graph backwards
            total_loss = recons_loss + sink_loss + src_cls_loss
            total_loss.backward()
            self.optimizer.step()

            losses = {
                'Src_cls_loss': src_cls_loss.item(),
                'Sink_loss': sink_loss.item(),
                'Recons_loss': recons_loss.item(),
            }

            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()

    def correction_epoch(self, src_loader, trg_loader, avg_meter):
        """Phase 2: Reconstruction-only fine-tuning (correct)."""
        self.feature_extractor.train()
        self.decoder.train()

        joint_loader = enumerate(zip(src_loader, trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, trg_x = src_x.to(self.device), trg_x.to(self.device)

            self.coptimizer.zero_grad()

            src_feat, out_s = self.feature_extractor(src_x)
            trg_feat, out_t = self.feature_extractor(trg_x)

            src_recon = self.decoder(src_feat, out_s)
            trg_recon = self.decoder(trg_feat, out_t)

            recons_loss = 1e-4 * (self.recons(trg_recon, trg_x) + self.recons(src_recon, src_x))
            recons_loss.backward()
            self.coptimizer.step()

            losses = {'Recons_loss': recons_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

    def update(self, src_loader, trg_loader, avg_meter, val_loader, logger):
        """
        Override base update() to implement RAINCOAT's two-phase training:
        Phase 1: classification + alignment + reconstruction (num_epochs)
        Phase 2: reconstruction-only correction (num_epochs)
        """
        best_src_risk = float('inf')
        best_model = None

        # === Phase 1: Main training ===
        logger.debug("===== RAINCOAT Phase 1: Train =====")
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for key in avg_meter.keys():
                avg_meter[key].reset()

            self.network.train()
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # Validation
            self.network.eval()
            val_loss_meter = AverageMeter()
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                    val_feat, _ = self.feature_extractor(val_x)
                    val_pred = self.classifier(val_feat)
                    val_loss = self.cross_entropy(val_pred, val_y)
                    val_loss_meter.update(val_loss.item(), val_x.size(0))

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Phase 1 Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')

            metrics_to_log = {key: val.avg for key, val in avg_meter.items()}
            mlflow.log_metrics(metrics_to_log, step=epoch)
            logger.debug(f'-------------------------------------')

        # === Phase 2: Correction ===
        logger.debug("===== RAINCOAT Phase 2: Correct =====")
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for key in avg_meter.keys():
                avg_meter[key].reset()

            self.correction_epoch(src_loader, trg_loader, avg_meter)

            # Validation after correction
            self.network.eval()
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                    val_feat, _ = self.feature_extractor(val_x)
                    val_pred = self.classifier(val_feat)
                    val_loss = self.cross_entropy(val_pred, val_y)
                    val_loss_meter.update(val_loss.item(), val_x.size(0))

            if (epoch + 1) % 10 == 0 and avg_meter.get('Src_cls_loss') and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Phase 2 Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')

            metrics_to_log = {f'correct_{key}': val.avg for key, val in avg_meter.items()}
            mlflow.log_metrics(metrics_to_log, step=self.hparams["num_epochs"] + epoch)
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()
        return last_model, best_model


# =====================================================================
# =========================  SSSS_TSA  ================================
# =====================================================================

class SSSS_TSA(Algorithm):
    """
    SSSS_TSA: Sensor-Specific Subspace learning with channel Selection for
    Time Series domain Adaptation.

    Architecture:
    - One backbone CNN per input channel (each processes 1 channel independently)
    - Per-channel classification + Sinkhorn alignment
    - Multihead attention for learned channel weighting/selection
    - Combined classification + Sinkhorn alignment on attention-weighted features
    - Total loss = combined losses + individual channel losses
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        true_final_out_channels = configs.final_out_channels  # per-channel feature dim
        self.true_input_channel = configs.input_channels

        # Override temp from hparams if provided
        if 'tau_temp' in hparams:
            configs.temp = hparams['tau_temp']

        # Override default feature_extractor with SepReps (per-channel backbones + attention)
        self.feature_extractor = SSSS_SepReps_with_multihead(configs, backbone)

        # Per-channel classifiers (input dim = per-channel feature dim * features_len)
        self.classifier_list_ind = nn.ModuleList([])
        for k in range(self.true_input_channel):
            self.classifier_list_ind.append(classifier(configs))

        # Combined classifier (input dim = per-channel dim * num_channels * features_len)
        configs.final_out_channels = true_final_out_channels * self.true_input_channel
        self.classifier = classifier(configs)
        configs.final_out_channels = true_final_out_channels  # restore

        # Rebuild network sequential for base class .train()/.eval() and state_dict
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # Combined optimizer (backbones + individual classifiers + attention + combined classifier)
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.backbone_nets.parameters()) +
            list(self.classifier_list_ind.parameters()) +
            list(self.feature_extractor.multihead_attention.parameters()) +
            list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.5, 0.99)
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # Sinkhorn distance for domain alignment
        sink_eps = hparams.get('sink_epsilon', 1e-3)
        self.sink = SinkhornDistance(sink_eps, max_iter=1000, reduction='sum')

        self.hparams = hparams
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # classifier_list_ind is not part of self.network, set train explicitly
        self.classifier_list_ind.train()

        joint_loader = enumerate(zip(src_loader, trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            self.optimizer.zero_grad()

            # === Per-channel losses ===
            src_reps_list = self.feature_extractor.fetch_individual_reps(src_x)
            trg_reps_list = self.feature_extractor.fetch_individual_reps(trg_x)

            clfr_src_loss = 0
            align_loss = 0
            for k in range(self.true_input_channel):
                # Per-channel classification
                clfr_k_pred = self.classifier_list_ind[k](src_reps_list[k])
                clfr_src_loss = clfr_src_loss + self.cross_entropy(clfr_k_pred.squeeze(), src_y)
                # Per-channel Sinkhorn alignment
                align_loss_k = self.sink(src_reps_list[k], trg_reps_list[k])[0]
                align_loss = align_loss + align_loss_k

            loss_ind = (self.hparams["src_cls_loss_wt"] * clfr_src_loss +
                        self.hparams["domain_loss_wt"] * align_loss)

            # === Combined channel losses (through attention, detached from backbone) ===
            comb_reps_src, _ = self.feature_extractor.combine_ind_through_attn(src_reps_list)
            clfr_pred_comb = self.classifier(comb_reps_src)
            loss_sup_src = self.cross_entropy(clfr_pred_comb.squeeze(), src_y)

            comb_reps_trg, _ = self.feature_extractor.combine_ind_through_attn(trg_reps_list)
            domain_loss = self.sink(comb_reps_src, comb_reps_trg)[0]

            # Total = combined + individual
            loss_total = (self.hparams["src_cls_loss_wt"] * loss_sup_src +
                          self.hparams["domain_loss_wt"] * domain_loss) + loss_ind

            loss_total.backward()
            self.optimizer.step()

            losses = {
                'Total_loss': loss_total.item(),
                'Src_cls_loss': loss_sup_src.item(),
                'Domain_loss': domain_loss.item(),
            }
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()


# =============================================================================
# =========================  CLUDA  ===========================================
# =============================================================================
class CLUDA(Algorithm):
    """
    CLUDA: Contrastive Learning for Unsupervised Domain Adaptation of Time Series.
    Uses a momentum-encoder, queue-based contrastive learning, nearest-neighbour
    cross-domain alignment, and adversarial domain discrimination.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # Replace the default feature_extractor with the CLUDA network
        self.feature_extractor = CLUDA_Network(backbone, configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.5, 0.99),
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device
        self.criterion_CL = nn.CrossEntropyLoss()

        # Augmenter will be created lazily on first batch
        self._augmenter = None
        self._input_channels = configs.input_channels

    # ---- lazy augmenter creation ----
    def _get_augmenter(self, seq_len, num_channels):
        cutout_len = max(1, seq_len // 12)
        if num_channels != 1:
            self._augmenter = CLUDA_Augmenter(cutout_length=cutout_len)
        elif seq_len > 1000:
            aug = CLUDA_Augmenter(cutout_length=cutout_len, cutout_prob=1,
                                   crop_min_history=0.25, crop_prob=1, dropout_prob=0.0)
            aug.augmentations = [aug.history_cutout, aug.history_cutout,
                                 aug.history_cutout, aug.history_crop,
                                 aug.gaussian_noise, aug.spatial_dropout]
            self._augmenter = aug
        else:
            self._augmenter = CLUDA_Augmenter(cutout_length=cutout_len, dropout_prob=0.0)

    def _augment(self, x):
        """Augment a (N, C, L) tensor: transpose to (N, L, C), augment, transpose back."""
        x_t = x.transpose(1, 2)                         # (N, L, C)
        mask = torch.ones_like(x_t)
        x_aug, _ = self._augmenter(x_t, mask)
        return x_aug.transpose(1, 2)                     # back to (N, C, L)

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, trg_loader))
        num_batches = min(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            # Lazily build augmenter on first batch
            if self._augmenter is None:
                seq_len = src_x.shape[2]   # (N, C, L)
                self._get_augmenter(seq_len, self._input_channels)

            p = float(step + epoch * num_batches) / (self.hparams["num_epochs"] * num_batches)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            # Create two augmented views of source and target
            q_src = self._augment(src_x)
            k_src = self._augment(src_x)
            q_trg = self._augment(trg_x)
            k_trg = self._augment(trg_x)

            self.optimizer.zero_grad()

            # Contrastive + domain discrimination forward
            (logits_s, labels_s, logits_t, labels_t,
             logits_ts, labels_ts, pred_domain, labels_domain,
             q_s) = self.feature_extractor.contrastive_update(
                q_src, k_src, q_trg, k_trg, alpha)

            # Contrastive losses
            loss_s = self.criterion_CL(logits_s, labels_s)
            loss_t = self.criterion_CL(logits_t, labels_t)
            loss_ts = self.criterion_CL(logits_ts, labels_ts)

            # Domain discrimination loss
            loss_disc = F.binary_cross_entropy(pred_domain, labels_domain)

            # Source classification loss
            pred_s = self.classifier(q_s)
            src_cls_loss = self.cross_entropy(pred_s, src_y)

            loss = loss_s + loss_t + loss_ts + loss_disc + src_cls_loss

            loss.backward()
            self.optimizer.step()

            losses = {
                'Total_loss': loss.item(),
                'Src_cls_loss': src_cls_loss.item(),
                'Domain_loss': loss_disc.item(),
            }
            for key, val in losses.items():
                avg_meter[key].update(val, src_x.size(0))

        self.lr_scheduler.step()