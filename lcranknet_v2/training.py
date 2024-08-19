import argparse
import os
import logging
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from LCRankNet import build_model
from optimizer import build_optimizer, build_scheduler
from utils import *
from data import *


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_pos, weight_neg):
        super(WeightedBCELoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def forward(self, predictions, targets):
        epsilon = 1e-6
        loss = self.weight_pos * (targets * torch.log(predictions+epsilon)) + \
               self.weight_neg * ((1 - targets) * torch.log(1 - predictions+epsilon))
        return -torch.mean(loss)

def get_p_hat(m1, m2):
    sigmoid = torch.nn.Sigmoid()
    p_hat = sigmoid(m1-m2)
    return p_hat

def get_p(m1, m2, dif_threshold):
    abs_diff = torch.abs(m1-m2)
    p = torch.zeros_like(m1)
    p[abs_diff <= dif_threshold] = 0.5
    p[m1>m2] = 1.0
    return p

def consistent(a1, a2, b1, b2, dif_threshold):
    # whether the rank between a1 and a2 is 
    # consistent with the rank between b1 and b2
    cond1 = torch.abs(a1-a2) <= dif_threshold
    cond2 = torch.abs(b1-b2) <= dif_threshold
    cond3 = (a1-a2)*(b1-b2) > 0
    return ((cond1&cond2)|cond3).int()
    
def get_pair_loss_v0(m1, m2, m_hat1, m_hat2, cfg):
    # rec loss + rank loss
    dif_threshold = cfg['training']['dif_threshold']
    rank_weight = cfg['training']['v0']['rank_weight']
    rec_weight = cfg['training']['v0']['rec_weight']

    p = get_p(m1, m2, dif_threshold) # m1 > m2
    p_hat = get_p_hat(m_hat1, m_hat2)
    p_hat = torch.clamp(p_hat, min=1e-7, max=1-1e-7)

    #ce_loss = -(p*torch.log(p_hat) + (1-p)*torch.log(1-p_hat)).mean()
    rank_loss = F.binary_cross_entropy(p_hat, p)
    mse_loss = nn.MSELoss()
    rec_loss = (mse_loss(m_hat1, m1) + mse_loss(m_hat2, m2))/2
    total_loss = rank_weight * rank_loss + rec_weight * rec_loss
    # average loss upon pair
    loss_dict = {"loss": total_loss,
                 "rec_loss": rec_loss,
                 "rank_loss": rank_loss}
    return loss_dict

def get_improv_loss(m, m_hat, m_bsf, cfg):
    p_imp = get_p_hat(m_bsf, m_hat) # sigmoid(m_bsf-m_hat): P(m_hat < m_bsf)
    improv_label = (m < m_bsf).float()
    loss_version = cfg['training']['loss']
    bce_weight = cfg['training'][loss_version]['bce_weight'] # positive: p(m<m_bsf)
    if cfg['data']['metric'] == "perplexity":
        bce_pos_weight = bce_weight
    else:
        bce_pos_weight = 1 - bce_weight
    bce_neg_weight = 1 - bce_pos_weight
    weighted_bce = WeightedBCELoss(bce_pos_weight, bce_neg_weight)
    improv_loss = weighted_bce(p_imp, improv_label)
    #improv_loss = F.binary_cross_entropy(p_imp, improv_label)
    return improv_loss

def get_improv_prob_loss(m, mu, sigma, m_bsf, cfg):
    epsilon = 1e-6
    distribution = Normal(mu, sigma+epsilon)
    p_hat = distribution.cdf(m_bsf) # mu < m_bsf
    improv_label = (m < m_bsf).float()
    bce_weight = cfg['training'][loss_version]['bce_weight'] # positive: p(m<m_bsf)
    if cfg['data']['metric'] == "perplexity":
        bce_pos_weight = bce_weight
    else:
        bce_pos_weight = 1 - bce_weight
    bce_neg_weight = 1 - bce_pos_weight
    weighted_bce = WeightedBCELoss(bce_pos_weight, bce_neg_weight)
    improv_loss = weighted_bce(p_imp, improv_label)
    #improv_loss = F.binary_cross_entropy(p_hat, improv_label)
    return improv_loss

def get_pair_loss_v1(m1, m2, m_hat1, m_hat2, m_bsf1, m_bsf2, cfg):
    # rec loss + rank loss + conti. improv. nll loss
    v0_loss_dict = get_pair_loss_v0(m1, m2, m_hat1, m_hat2, cfg)
    improv_loss1 = get_improv_loss(m1, m_hat1, m_bsf1, cfg)
    improv_loss2 = get_improv_loss(m2, m_hat2, m_bsf2, cfg)
    improv_weight = cfg['training']['v1']['improv_weight']
    improv_loss = (improv_loss1 + improv_loss2)/2
    rank_weight = cfg['training']['v1']['rank_weight']
    rec_weight = cfg['training']['v1']['rec_weight']
    rank_loss = v0_loss_dict['rank_loss']
    rec_loss = v0_loss_dict['rec_loss']
    total_loss = improv_loss * improv_weight + rank_loss * rank_weight + rec_loss * rec_weight
    loss_dict = {"loss": total_loss,
                 "rec_loss": rec_loss,
                 "rank_loss": rank_loss,
                 "improv_loss": improv_loss}
    return loss_dict

def gaussian_nll_loss(m, mu, sigma):
    # compute the Gaussian Negative Log-Likelihood loss
    #epsilon = 1e-6 # to prevent division by zero
    #loss = 0.5 * torch.log(2*torch.pi*sigma**2+epsilon) + (m-mu)**2 / (2*sigma**2+epsilon)
    gnll = nn.GaussianNLLLoss()
    loss = gnll(mu, m, sigma)
    return loss

def get_pair_loss_v2(m1, m2, mu1, mu2, sigma1, sigma2, cfg):
    # rec prob loss + rank prob loss
    rec_loss = (gaussian_nll_loss(m1, mu1, sigma1) + gaussian_nll_loss(m2, mu2, sigma2)) / 2
    rec_loss = torch.mean(rec_loss)
    rec_weight = cfg['training']['v2']['rec_weight']
    
    dif_threshold = cfg['training']['dif_threshold']
    p = get_p(m1, m2, dif_threshold) # m1>m2
    mu_diff = mu1 - mu2
    epsilon = 1e-6
    sigma_diff = torch.sqrt(sigma1**2+sigma2**2) + epsilon
    mu_diff = torch.clamp(mu_diff, min=-1e4, max=1e4)
    sigma_diff = torch.clamp(sigma_diff, min=epsilon, max=1e3)
    diff_distribution = Normal(mu_diff, sigma_diff)
    # p(mu1>mu2)
    p_hat = 1 - diff_distribution.cdf(torch.zeros_like(mu1))
    rank_loss = F.binary_cross_entropy(p_hat, p)
    rank_weight = cfg['training']['v2']['rank_weight']
    total_loss = rec_weight*rec_loss + rank_weight*rank_loss
    loss_dict = {"loss": total_loss,
                 "rec_loss": rec_loss,
                 "rank_loss": rank_loss}
    return loss_dict

def get_pair_loss_v3(m1, m2, mu1, mu2, sigma1, sigma2, m_bsf1, m_bsf2, cfg):
    # rec prob loss + rank prob loss + conti.improv. prob loss
    improv_weight = cfg['training']['v3']['improv_weight']
    rec_weight = cfg['training']['v3']['rec_weight']
    rank_weight = cfg['training']['v3']['rank_weight']
    v2_loss_dict = get_pair_loss_v2(m1, m2, mu1, mu2, sigma1, sigma2, cfg)
    improv_loss1 = get_improv_prob_loss(m1, mu1, sigma1, m_bsf1, cfg)
    improv_loss2 = get_improv_prob_loss(m2, mu2, sigma2, m_bsf2, cfg)
    improv_loss = (improv_loss1 + improv_loss2)/2
    rec_loss = v2_loss_dict['rec_loss']
    rank_loss = v2_loss_dict['rank_loss']
    total_loss = improv_loss * improv_weight + rec_weight * rec_loss + rank_weight * rank_loss
    loss_dict = {"loss": total_loss,
                 "rec_loss": rec_loss,
                 "rank_loss": rank_loss,
                 "improv_loss": improv_loss}
    return loss_dict

def get_confusion_matrix_v0(m1, m2, m_hat1, m_hat2, m_bsf1, m_bsf2, cfg):
    with torch.no_grad():
        dif_threshold = cfg['training']['dif_threshold']
        # bsf: best performance so far
        correct_counts = consistent(m1, m2, m_hat1, m_hat2, dif_threshold)
        bsf_counts = consistent(m1, m2, m_bsf1, m_bsf2, dif_threshold)
        bsf_confusion_matrix = torch.zeros((len(m1), 4))
        # BB = ['TT', 'FT', 'TF', 'FF'] 
        # first: bsf == optimal 
        # second: pred == optimal
        bsf_confusion_matrix[:, 0] = (bsf_counts == 1) & (correct_counts == 1)
        bsf_confusion_matrix[:, 1] = (bsf_counts == 0) & (correct_counts == 1)
        bsf_confusion_matrix[:, 2] = (bsf_counts == 1) & (correct_counts == 0)
        bsf_confusion_matrix[:, 3] = (bsf_counts == 0) & (correct_counts == 0)
        bsf_confusion_vector = torch.sum(bsf_confusion_matrix, 0)
        correct_num = torch.sum(correct_counts).int()
    return correct_num, bsf_confusion_vector

def get_confusion_matrix_v1(m1, m2, m_hat1, m_hat2, m_bsf1, m_bsf2, cfg):
    with torch.no_grad():
        p_imp1 = get_p_hat(m_bsf1, m_hat1) # sigmoid(m_bsf-m_hat): P(m_hat < m_bsf)
        p_imp2 = get_p_hat(m_bsf2, m_hat2) # sigmoid(m_bsf-m_hat): P(m_hat < m_bsf)
        if cfg['data']['metric'] == 'bleu':
            p_imp1 = 1 - p_imp1
            p_imp2 = 1 - p_imp2
        improv_threshold = cfg['training']['v1']['improv_threshold']
        m_hat_updated1 = torch.where(p_imp1 > improv_threshold, m_hat1, m_bsf1)
        m_hat_updated2 = torch.where(p_imp2 > improv_threshold, m_hat2, m_bsf2)
        correct_num, bsf_confusion_vector = get_confusion_matrix_v0(m1, m2, \
            m_hat_updated1, m_hat_updated2, \
            m_bsf1, m_bsf2, cfg)
    return correct_num, bsf_confusion_vector

def get_confusion_matrix_v23(m1, m2, mu1, mu2, sigma1, sigma2, m_bsf1, m_bsf2, cfg):
    with torch.no_grad():
        mu_diff = mu1 - mu2
        epsilon = 1e-6
        sigma_diff = torch.sqrt(sigma1**2+sigma2**2) + epsilon
        diff_distribution = Normal(mu_diff, sigma_diff)
        # p(mu1>mu2)
        p_rank_hat = 1 - diff_distribution.cdf(torch.zeros_like(mu1))
        rank_threshold = cfg['training'][cfg['training']['loss']]['rank_threshold']
        #if not (p_rank_hat >= rank_threshold or p_rank_hat <= 1-rank_threshold):
        #    m_hat1 = m_bsf1
        #    m_hat2 = m_bsf2
        mu_updated1 = torch.where((p_rank_hat >= rank_threshold) | \
            (p_rank_hat <= 1-rank_threshold), mu1, m_bsf1)
        mu_updated2 = torch.where((p_rank_hat >= rank_threshold) | \
            (p_rank_hat <= 1-rank_threshold), mu2, m_bsf2)
        correct_num, bsf_confusion_vector = get_confusion_matrix_v0(m1, m2, \
            mu_updated1, mu_updated2, m_bsf1, m_bsf2, cfg)
    return correct_num, bsf_confusion_vector

def get_batch_loss(batch, mu, sigma, cfg, step=0):
    # get loss
    optimals = torch.tensor(batch['optimals']).to(mu.device)
    batch_size = optimals.size(0)
    m_hat1, m_hat2 = mu[0::2], mu[1::2]
    sigma1, sigma2 = sigma[0::2], sigma[1::2]
    m_hat1 = m_hat1.squeeze()
    m_hat2 = m_hat2.squeeze()
    m1, m2 = optimals[0::2], optimals[1::2]
    # if step % 500 == 0 and step > 1:
    #     import pdb; pdb.set_trace()
    # get confusion matrix
    curves = torch.stack(batch['curves']).to(mu.device)
    if cfg['data']['metric'] == "perplexity":
        curves[curves == 0] = float('inf')
        curve_bsfs = torch.min(curves, dim=1).values
    elif cfg['data']['metric'] == "bleu":
        curves[curves == 0] = float('-inf')
        curve_bsfs = torch.max(curves, dim=1).values
    m_bsf1, m_bsf2 = curve_bsfs[0::2], curve_bsfs[1::2]
    if cfg['training']['loss'] == 'v0':
        loss_dict = get_pair_loss_v0(m1, m2, m_hat1, m_hat2, cfg)
        correct_num, bsf_confusion_vector = get_confusion_matrix_v0(m1, m2, m_hat1, m_hat2, m_bsf1, m_bsf2, cfg)
    elif cfg['training']['loss'] == 'v1':
        loss_dict = get_pair_loss_v1(m1, m2, m_hat1, m_hat2, m_bsf1, m_bsf2, cfg)
        correct_num, bsf_confusion_vector = get_confusion_matrix_v1(m1, m2, m_hat1, m_hat2, m_bsf1, m_bsf2, cfg)
    elif cfg['training']['loss'] == 'v2':
        loss_dict = get_pair_loss_v2(m1, m2, m_hat1, m_hat2, sigma1, sigma2, cfg)
        correct_num, bsf_confusion_vector = get_confusion_matrix_v23(m1, m2, m_hat1, m_hat2, sigma1, sigma2, m_bsf1, m_bsf2, cfg)
    elif cfg['training']['loss'] == 'v3':
        loss_dict = get_pair_loss_v3(m1, m2, m_hat1, m_hat2, sigma1, sigma2, m_bsf1, m_bsf2, cfg)
        correct_num, bsf_confusion_vector = get_confusion_matrix_v23(m1, m2, m_hat1, m_hat2, sigma1, sigma2, m_bsf1, m_bsf2, cfg)
    return loss_dict, correct_num, bsf_confusion_vector 
    
def print_confusion_vector(confusion_vector):
    #keys = ['TT', 'FT', 'TF', 'FF'] 
    output = "BSF O Pred O: " + str(int(confusion_vector[:,0].item()))
    output += "\tBSF X Pred O: "  + str(int(confusion_vector[:,1].item()))
    output += "\nBSF O Pred X: " + str(int(confusion_vector[:,2].item()))
    output += "\tBSF X Pred X: " + str(int(confusion_vector[:,3].item()))
    return output

def evaluation(model, dataloader, cfg):
    model.eval()
    total_correct_num = 0
    total_bsf_confusion_vecor = torch.zeros((1, 4))
    total_loss_dict = {}
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            mu, sigma = model(**batch)
            loss_dict, correct_num, bsf_confusion_vector = get_batch_loss(batch, mu, sigma, cfg)
            for k in loss_dict:
                if k not in total_loss_dict:
                    total_loss_dict[k] = loss_dict[k].item()
                else:
                    total_loss_dict[k] += loss_dict[k].item()
            total_correct_num += correct_num
            total_bsf_confusion_vecor += bsf_confusion_vector
    return total_loss_dict, total_correct_num, total_bsf_confusion_vecor

def present_loss(loss_dict, num_batches):
    loss_names = list(loss_dict.keys())
    out_str = ""
    for name in loss_names:
        out_str += name + ": " + "{:.2e}".format(loss_dict[name]/num_batches)  + " "
    out_str += "\n"
    return out_str

def present_train_results(epoch_no, step, num_train_batches, batch_size,
                        train_loss_dict, train_correct_num, train_bsf_confusion_vector, logger):
    printed = "Epoch: {0}, Step: {1}/{2} ({3:.2f}%), [TRAIN] Accuracy (ranking between two curves): {4:.2f}%\n"
    formated_printed = printed.format(epoch_no, step, num_train_batches, \
                step/num_train_batches * 100, \
                train_correct_num/(num_train_batches*(batch_size//2))*100)
    loss_str = present_loss(train_loss_dict, num_train_batches)
    formated_printed += loss_str
    formated_printed += "Best performance so far vs. predictions: \n" + \
        print_confusion_vector(train_bsf_confusion_vector)
    logger.info(formated_printed)

def present_dev_results(num_dev_batches, batch_size, dev_loss_dict, 
                        dev_correct_num, dev_bsf_confusion_vector,
                        best_score, ckpt_dir, best_ckpt, model, 
                        learning_rate, logger):
    formated_printed = '[DEV] Accuracy (ranking between two curves): {0:.2f}%\n'.format(
                dev_correct_num/(num_dev_batches*(batch_size//2))*100)
    loss_str = present_loss(dev_loss_dict, num_dev_batches)
    formated_printed += loss_str
    formated_printed += "Best performance so far vs. predictions: \n" + \
        print_confusion_vector(dev_bsf_confusion_vector)
    logger.info(formated_printed)
    logger.info("Learning rate: " + str(learning_rate) + "\n")
    if dev_correct_num > best_score:
        best_score = dev_correct_num
        logger.info("Save best checkpoint so far to "+ ckpt_dir)
        torch.save(model.to('cpu').state_dict(), best_ckpt)
        with open(os.path.join(ckpt_dir, "best_ckpt.eval"), 'w') as f:
            f.write(formated_printed)
    return best_score, formated_printed

def get_logger(log_file):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Learning curve prediction.")
    parser.add_argument(
        "--config",
        default="scratch_configs.yaml",
        type=str,
        help="Configuration file (yaml).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(seed=cfg['training']['random_seed'])
    model = build_model(cfg)
    model_dir = cfg['model_dir']
    ckpt_dir = os.path.join(model_dir, cfg['model_name'])
    best_ckpt = os.path.join(ckpt_dir, "best_ckpt.pth")
    if cfg['training']['load_ckpt'] and os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt))
    else:
        os.makedirs(os.path.join(ckpt_dir), exist_ok=True)
    logger = get_logger(log_file=os.path.join(ckpt_dir, "training.log"))
    shutil.copy(args.config, os.path.join(ckpt_dir, "config.yaml"))
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Model directory: {}'.format(ckpt_dir))
    logger.info('#Total parameters = {}'.format(total_params))
    logger.info('#Total trainable parameters = {}'.format(total_params_trainable))
    model.to(cfg['device'])
    train_dataloader, train_sampler = build_dataloader(cfg, build_dataset(cfg['data'], 'train', logger))
    dev_dataloader, dev_sampler = build_dataloader(cfg, build_dataset(cfg['data'], 'dev', logger), 'dev')
    num_dev_batches = len(dev_dataloader)
    num_train_batches = len(train_dataloader)
    batch_size = cfg['training']['batch_size']
    batch_size = batch_size if batch_size%2==0 else 2*batch_size
    num_train_samples = int(num_train_batches * (batch_size//2))
    num_dev_samples = int(num_dev_batches * (batch_size//2))
    logger.info("#Train examples: {0}, #Dev examples: {1}".format(num_train_samples, num_dev_samples))
    optimizer = build_optimizer(config=cfg['training'], model=model)#.module)
    scheduler, scheduler_type = build_scheduler(cfg['training'], optimizer=optimizer)
    val_unit, val_freq = cfg['training']['validation']['unit'], cfg['training']['validation']['freq']
    #best_score = sys.maxsize
    best_score = 0
    best_printed = ""
    global_step = 0
    train_total_correct_num = 0
    num_train_batches_eval = 0
    for epoch_no in range(cfg['training']['total_epoch']):
        train_loss_dict = {}
        train_total_bsf_confusion_vector = torch.zeros((1, 4))
        for step, batch in enumerate(train_dataloader):
            num_train_batches_eval += 1
            if val_unit=="step" and global_step%val_freq==0 and global_step>0:
                dev_loss_dict, dev_correct_num, dev_bsf_confusion_vector = \
                    evaluation(model, dev_dataloader, cfg)
                learning_rate = optimizer.param_groups[0]['lr']
                present_train_results(epoch_no, step, num_train_batches_eval, batch_size,
                        train_loss_dict, train_total_correct_num, train_total_bsf_confusion_vector, logger)
                num_train_batches_eval = 0
                train_total_correct_num = 0
                best_score, best_printed = present_dev_results(num_dev_batches, batch_size, dev_loss_dict, 
                        dev_correct_num, dev_bsf_confusion_vector,
                        best_score, ckpt_dir, best_ckpt, model, 
                        learning_rate, logger)
                model.to(cfg['device'])
            model.train()
            mu, sigma = model(**batch)
            loss_dict, correct_counts, bsf_confusion_vector = get_batch_loss(batch, mu, sigma, cfg, step)
            pre_batch = batch
            loss = loss_dict['loss']
            '''
            for name, param in model.named_parameters():
                print(f"{name} max: {param.data.max()}")
                print(f"{name} min: {param.data.min()}")
            '''
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            '''
            # Monitor gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} gradient max: {param.grad.data.max()}")
                    print(f"{name} gradient min: {param.grad.data.min()}")
            '''
            for k in loss_dict:
                train_loss_dict[k] = train_loss_dict.get(k, 0) + loss_dict[k].item()
            train_total_correct_num += correct_counts
            train_total_bsf_confusion_vector += bsf_confusion_vector
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1