"""
Author: Benny
Date: Nov 2019
"""
import time
from dataset import ModelNetDataLoader, ScanObjectNN
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf
from spikingjelly.clock_driven import functional

def test(model, loader, num_class=40, vote_num=100):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    is_scanobjectnn = True if num_class == 15 else False
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.cuda(), target.cuda()
        target = target if is_scanobjectnn else target[:, 0]
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        functional.reset_net(classifier)
        for _ in range(vote_num):
            pred = classifier(points)
            functional.reset_net(classifier)
            vote_pool += pred

        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

def test_novote(model, loader, num_class=40):
    classifier = model.eval()
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    if num_class == 15: is_scanobjectnn = True 
    else: is_scanobjectnn = False
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target if is_scanobjectnn  else target[:, 0]  ## 
        points, target = points.cuda(), target.cuda()

        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        functional.reset_net(classifier)
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))

    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join([str(g) for g in args['gpu']]) 
    logger = logging.getLogger(__name__)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    root = hydra.utils.to_absolute_path(args.dataset.DATA_PATH)
    if args.dataset.name == 'ModelNet':        
        TRAIN_DATASET = ModelNetDataLoader(root=root, nclass=args.dataset.num_class, npoint=args.num_point, split='train', normal_channel=args.dataset.normal)
        TEST_DATASET = ModelNetDataLoader(root=root, nclass=args.dataset.num_class, npoint=args.num_point, split='test', normal_channel=args.dataset.normal)
    elif args.dataset.name == 'ScanObjectNN':
        TRAIN_DATASET = ScanObjectNN(root=root, num_points=args.num_point, split='training')
        TEST_DATASET = ScanObjectNN(root=root, num_points=args.num_point, split='test')
    else: raise NotImplementedError(f'{args.name} dataset is not found')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=6)

    '''MODEL LOADING'''
    args.num_class = 40
    args.input_dim = 6 if args.dataset.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs, which is {args['gpu']}")
        classifier = torch.nn.DataParallel(classifier)
        classifier.to(torch.device('cuda:0'))
    else:
        logger.info(f"Using {torch.cuda.device_count()} GPU, which is {args['gpu']}")
        classifier.to(torch.device('cuda:0'))

    try:
        checkpoint = torch.load('best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model...')
    
    '''TESTING'''
    logger.info('Start Evaluation...')


    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, args.num_class)

        logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))

    logger.info('End of Evaluation...')

if __name__ == '__main__':
    main()
