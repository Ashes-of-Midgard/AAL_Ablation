import os
import sys
import numpy as np
import time
import scipy.stats
import torch
from torch.utils.data import DataLoader
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import scipy
# import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from attack_type import attackers
from utils import CrossEntropyLabelSmooth
from torch.autograd import Variable
from CBAM_img_5 import resnet50_cbam as resnet50
from CBAM_img_5 import ResNet


parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='Cri2_ImageNet_Best', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
# parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--nb_iter', help='Adversarial attack iteration', type=int, default=3)
parser.add_argument('--eps', help='Adversarial attack maximal perturbation', type=float, default=8)
parser.add_argument('--eps_iter', help='Adversarial attack step size', type=float, default=2/255)
parser.add_argument('--initial_const', help='initial value of the constant c', type=float, default=0.1)
parser.add_argument('--attack_type', help='type of adversarial attack', type=str, default='')
parser.add_argument('--label_smooth', type=float, default=0, help='label smoothing')
parser.add_argument('--adv_epoch', type=int, default=0, help='epoch of adv training') # 200
parser.add_argument('--threshold', type=float, default=0.5, help='label smoothing')
parser.add_argument('--back_epoch', type=float, default=20, help='epoch of backtracking')
parser.add_argument('--back', type=int, default=0, help='epoch of backtracking')
parser.add_argument('--back_rate', type=float, default=0.01, help='back tracking rate')
parser.add_argument('--gpu', default="0,1,2,3", help='gpu device id')
parser.add_argument('--resume', '-r', type=str, default='', help='resume from checkpoint') # action='store_true',
args, unparsed = parser.parse_known_args()
args.gpu = [int(i) for i in args.gpu.split(',')]
args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 10

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    # num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(args.gpu[0])
    # genotype = eval("genotypes.%s" % args.arch)
    # print('---------Genotype---------')
    # logging.info(genotype)
    print('--------------------------')
    att = False
    sw = []
    model = resnet50(flag=att,sw=sw)
    if len(args.gpu)> 1:
        # model = nn.DataParallel(model)
        # model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )


    if args.resume:
        logging.info('==> Resuming from checkpoint..')
        logging.info(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_adv_top1 = checkpoint['best_acc_top1']
        # if not args.tuning:
        start_epoch = checkpoint['epoch']
        logging.info("Current best Prec@1 = {:.4%}".format(best_adv_top1/100))
    else:
        logging.info('==> ORI Training')
        start_epoch = 0

    # data_dir = os.path.join(args.tmp_data_dir, 'imagenet_search')
    # traindir = os.path.join(data_dir, 'train')
    # validdir = os.path.join(data_dir, 'val')
    traindir = os.path.join(args.tmp_data_dir, 'train_10')
    validdir = os.path.join(args.tmp_data_dir, 'val_10')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    
    print('Data loaded successfully!!')


    train_criterion = CrossEntropyLabelSmooth(CLASSES, args.label_smooth).cuda() \
        if args.label_smooth > 0 else criterion
    
    print('loss function initialization succeed!!')
    test_criterion = criterion
    if args.attack_type:
        attacker_train = attackers[args.attack_type](model, loss_fn=train_criterion, eps=args.eps, nb_iter=args.nb_iter, \
                                                    eps_iter=args.eps_iter, num_classes=CLASSES, initial_const=args.initial_const)
        attacker_test = attackers['fgsm_delta'](model, loss_fn=test_criterion, eps=args.eps, nb_iter=args.nb_iter,\
                                                    eps_iter=args.eps_iter, num_classes=CLASSES, initial_const=args.initial_const)
        # attacker_test = attackers['pgd'](model, loss_fn=test_criterion, eps=args.eps, nb_iter=args.nb_iter, \
        #                                             eps_iter=args.eps_iter, num_classes=CLASSES, initial_const=args.initial_const)
        # if args.nb_iter == 2:
        #     attacker_test_7 = attackers['pgd'](model, loss_fn=test_criterion, eps=args.eps, nb_iter=7, \
        #                                             eps_iter=0.0078, num_classes=CLASSES, initial_const=args.initial_const)
    else:
        attacker_train = None
        attacker_test = None


#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc_top1 = 0
    best_acc_top5 = 0
    train_best = 0
    val_best = 0
    val_adv_best = 0
    for i in range(start_epoch):
        scheduler.step()
    for epoch in range(start_epoch, args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
        if len(args.gpu)> 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer,attacker_train,epoch)  #train
        logging.info('Train_acc: %f', train_acc)
        if train_acc > train_best:
            train_best = train_acc
        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)   #valid
        if valid_acc_top1 > val_best:
            val_best = valid_acc_top1
        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        # logging.info('Valid_acc_top5: %f', valid_acc_top5)
        valid_acc_top1_adv, valid_acc_top5_adv, valid_obj_adv, dists = infer_adv(valid_queue, model, criterion,attacker_test,epoch=epoch)   #valid_adv
        logging.info('dist_pixel_adv: %f', dists[0])
        logging.info('dist_pixel_adv_sa: %g', dists[1])
        logging.info('dist_feature_adv_eu: %f', dists[2])
        logging.info('dist_feature_adv_sa_eu: %f', dists[3])
        logging.info('dist_feature_adv_kl: %f', dists[4])
        logging.info('dist_feature_adv_sa_kl: %f', dists[5])
        logging.info('dist_feature_adv_emd: %f', dists[6])
        logging.info('dist_feature_adv_sa_emd: %f', dists[7])
        if valid_acc_top1_adv > val_adv_best:
            val_adv_best = valid_acc_top1_adv
            is_best = True
        logging.info('Valid_acc_top1_adv: %f', valid_acc_top1_adv)
        # logging.info('Valid_acc_top5_adv: %f', valid_acc_top5_adv)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)
        if (epoch+1) % 10 == 0:
            print ('train_best,val_best,val_adv_best:',train_best,val_best,val_adv_best)
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': val_adv_best,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)

def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs - epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(train_queue, model, criterion, optimizer,attacker,epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        b_start = time.time()
        optimizer.zero_grad()
        sw = []
        att = False
        # logits,sa_w,sa_wg = model(input)
        model.sw = sw
        model.flag = att

        logits,sa_w = model(input)

        sw = sa_w
        # zero = torch.zeros_like(sa_w)
        one = torch.ones_like(sa_w)
        # # zero = torch.zeros(shape[0], 3, shape[2],shape[3]).cuda()
        # # one = torch.ones(shape[0], 3, shape[2],shape[3]).cuda()
        # # sa_b = torch.where(sa_w > args.threshold, zero, one)   #train_adv
        # # sa_b = one - sa_w
        # # sa_b = pow(sa_b,2)
        # # sa_b = one
        # sa_b = zero      #train_ori

        loss = criterion(logits, target)
        # import pdb;pdb.set_trace()
        loss.backward()



################################backtrack######################################
        # if step >= args.back_epoch and args.back == 1:
        #     # sa_wg = model.module.sa.conv1.weight.grad
        #     back_label = torch.where(sa_w < args.threshold-0.1, zero, sa_w)
        #     back_label = torch.where(back_label > args.threshold, zero, back_label)
        #     back = torch.where(back_label > 0, sa_w, zero)
        #     # back = back_label * sa_wg
        #     sa_w = sa_w + back * 0.05
################################################################################


        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs',
                                    step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)


        if attacker != None and epoch >= args.adv_epoch:
            att = True
            # zero = torch.zeros_like(sa_w)
            # one = torch.ones_like(sa_w)
            # zero = torch.zeros(shape[0], 3, shape[2],shape[3]).cuda()
            # one = torch.ones(shape[0], 3, shape[2],shape[3]).cuda()      
            # sa_b = torch.where(sa_w > args.threshold, zero, one)
            # sa_b = torch.mul(sa_w,sa_w)
            # sa_b = zero
            adv,max_ten = attacker.perturb(back_rate=args.back_rate ,x=input, y=target, sa_b=sa_w)
            sw = (one - 0.05 * max_ten) * sw
            
            input= adv.cuda()
            C_start = time.time()
            optimizer.zero_grad()
            model.sw = sw.detach()
            model.flag = att
            logits,sa_w = model(input)
            
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_time.update(time.time() - C_start)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                end_time = time.time()
                if step == 0:
                    duration = 0
                    start_time = time.time()
                else:
                    duration = end_time - start_time
                    start_time = time.time()
                logging.info('TRAIN_adv Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs',
                                        step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            model.flag = False
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    return top1.avg, top5.avg, objs.avg

def infer_adv(valid_queue, model:ResNet, criterion,attacker, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        # adv = attacker.perturb(input, target)
        ### Ablation Modified ###
        dist_pixel_adv_sa_mean = 0.0
        dist_pixel_adv_mean = 0.0
        dist_feature_adv_eu_mean = 0.0
        dist_feature_adv_sa_eu_mean = 0.0
        dist_feature_adv_kl_mean = 0.0
        dist_feature_adv_sa_kl_mean = 0.0
        dist_feature_adv_emd_mean = 0.0
        dist_feature_adv_sa_emd_mean = 0.0
        with torch.no_grad():
            logits, sa_w = model(input)
        adv_sa = attacker.perturb(x=input, y=target, sa_b=sa_w)
        adv = attacker.perturb(x=input, y=target)
        dist_pixel_adv_sa = torch.dist(adv_sa, input) / len(input)
        dist_pixel_adv = torch.dist(adv, input) / len(input)
        #feature_ori = model(input)[0]
        #feature_adv_sa = model(adv_sa)[0]
        #feature_adv = model(adv)[0]
        feature_ori = F.softmax(model(input)[0], dim=1)
        feature_adv_sa = F.softmax(model(adv_sa)[0], dim=1)
        feature_adv = F.softmax(model(adv)[0], dim=1)
        dist_feature_adv_eu = torch.dist(feature_ori, feature_adv) / len(input)
        dist_feature_adv_sa_eu = torch.dist(feature_ori, feature_adv_sa) / len(input)
        dist_feature_adv_kl = F.kl_div(feature_ori, feature_adv,  reduction='batchmean')
        dist_feature_adv_sa_kl = F.kl_div(feature_ori, feature_adv_sa, reduction='batchmean')
        # Calculate EMD
        dist_feature_adv_emd = EMD(feature_ori, feature_adv)
        dist_feature_adv_sa_emd = EMD(feature_ori, feature_adv_sa)
        #dist_feature_adv_sa = F.cross_entropy(feature_ori, feature_adv_sa)
        #dist_feature_adv = F.cross_entropy(feature_ori, feature_adv)
        dist_pixel_adv_sa_mean += dist_pixel_adv_sa
        dist_pixel_adv_mean += dist_pixel_adv
        dist_feature_adv_eu_mean += dist_feature_adv_eu
        dist_feature_adv_sa_eu_mean += dist_feature_adv_sa_eu
        dist_feature_adv_kl_mean += dist_feature_adv_kl
        dist_feature_adv_sa_kl_mean += dist_feature_adv_sa_kl
        dist_feature_adv_emd_mean += dist_feature_adv_emd
        dist_feature_adv_sa_emd_mean += dist_feature_adv_sa_emd
        ### End Modified ###
        adv = attacker.perturb(x=input, y=target)
        input= adv.cuda()
        target = target.cuda(non_blocking=True)
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            model.flag = False
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID_adv Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    dist_pixel_adv_mean = dist_pixel_adv_mean / len(valid_queue)
    dist_pixel_adv_sa_mean = dist_pixel_adv_sa_mean / len(valid_queue)
    dist_feature_adv_eu_mean = dist_feature_adv_eu_mean / len(valid_queue)
    dist_feature_adv_sa_eu_mean = dist_feature_adv_sa_eu_mean / len(valid_queue)
    dist_feature_adv_kl_mean = dist_feature_adv_kl_mean / len(valid_queue)
    dist_feature_adv_sa_kl_mean = dist_feature_adv_sa_kl_mean / len(valid_queue)
    dist_feature_adv_emd_mean = dist_feature_adv_emd_mean / len(valid_queue)
    dist_feature_adv_sa_emd_mean = dist_feature_adv_sa_emd_mean / len(valid_queue)
    return top1.avg, top5.avg, objs.avg, (dist_pixel_adv_mean, dist_pixel_adv_sa_mean,
                                          dist_feature_adv_eu_mean, dist_feature_adv_sa_eu_mean,
                                          dist_feature_adv_kl_mean, dist_feature_adv_sa_kl_mean,
                                          dist_feature_adv_emd_mean, dist_feature_adv_sa_emd_mean)


def EMD(h1_batch:torch.Tensor, h2_batch:torch.Tensor):
    emd_mean = torch.zeros([1])
    for i in range(len(h1_batch)):
        h1 = np.array(h1_batch[i].detach().cpu())
        h2 = np.array(h2_batch[i].detach().cpu())
        location1 = np.array(range(len(h1)))
        location2 = np.array(range(len(h2)))
        emd_mean += scipy.stats.wasserstein_distance(h1, h2, location1, location2)
    return emd_mean / len(h1_batch)


if __name__ == '__main__':
    main()
