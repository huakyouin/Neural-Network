from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
import pickle
import yaml
import opts
# from misc.rewards import get_self_critical_reward
import torchvision.transforms as transforms
import pdb
import json
from misc import utils, eval_utils, AttModel
try:
    import tensorflow as tf
except ImportError:
    print("Magi_ZZ_ML_Kernel:>> Tensorflow not installed; No tensorboard logging.")
    tf = None

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(epoch, opt):
    model.train()
    #########################################################################################
    # Training begins here
    #########################################################################################
    data_iter = iter(dataloader)
    lm_loss_temp = 0
    bn_loss_temp = 0
    fg_loss_temp = 0
    cider_temp = 0
    rl_loss_temp = 0
    start = time.time()
    #mycount = 0
    #mybatch = 5
    #loss = 0
    for step in range(len(dataloader)-1):
        global iteration
        iteration+=1
        data = data_iter.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id = data
        proposals = proposals[:,:max(int(max(num[:,1])),1),:]
        bboxs = bboxs[:,:int(max(num[:,2])),:]
        box_mask = box_mask[:,:,:max(int(max(num[:,2])),1),:]

        input_imgs.data.resize_(img.size()).copy_(img)
        input_seqs.data.resize_(iseq.size()).copy_(iseq)
        gt_seqs.data.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.data.resize_(num.size()).copy_(num)
        input_ppls.data.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.data.resize_(bboxs.size()).copy_(bboxs)
        mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)
        loss = 0
        #model.init_hidden()
        #if mycount == 0:
        #model.zero_grad()
        #mycount = mybatch

        #If using RL for self critical sequence training
        if opt.self_critical:
            rl_loss, bn_loss, fg_loss, cider_score = model(input_imgs, input_seqs, gt_seqs, input_num, input_ppls, gt_bboxs, mask_bboxs, 'RL')
            cider_temp += cider_score.sum().data[0] / cider_score.numel()
            loss += (rl_loss.sum() + bn_loss.sum() + fg_loss.sum()) / rl_loss.numel()
            rl_loss_temp += loss.data[0]

        #If using MLE
        else:
            lm_loss, bn_loss, fg_loss = model(input_imgs, input_seqs, gt_seqs, input_num, input_ppls, gt_bboxs, mask_bboxs, 'MLE')
            loss += ((lm_loss.sum() + bn_loss.sum() + fg_loss.sum()) / lm_loss.numel())

            lm_loss_temp += (lm_loss.sum().data.item() / lm_loss.numel())
            bn_loss_temp += (bn_loss.sum().data.item() / lm_loss.numel()) 
            fg_loss_temp += (fg_loss.sum().data.item() / lm_loss.numel())

        model.zero_grad()
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        #utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        if opt.finetune_cnn:
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()

        if step % opt.disp_interval == 0 and step != 0:
            end = time.time()
            lm_loss_temp /= opt.disp_interval
            bn_loss_temp /= opt.disp_interval
            fg_loss_temp /= opt.disp_interval
            rl_loss_temp /= opt.disp_interval
            cider_temp /= opt.disp_interval

            print("step {}/{} (epoch {}), lm_loss = {:.3f}, bn_loss = {:.3f}, fg_loss = {:.3f}, rl_loss = {:.3f}, cider_score = {:.3f}, lr = {:.5f}, time/batch = {:.3f}" \
                .format(step, len(dataloader), epoch, lm_loss_temp, bn_loss_temp, fg_loss_temp, rl_loss_temp, cider_temp, opt.learning_rate, end - start))
            
            start = time.time()

            lm_loss_temp = 0
            bn_loss_temp = 0
            fg_loss_temp = 0
            cider_temp = 0
            rl_loss_temp = 0

        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', loss.item(), iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.learning_rate, iteration)
                tf_summary_writer.flush()
        # Write the training loss summary
        #if opt.self_critical:
        #    if (iteration % opt.losses_log_every == 0):
        #        if tf is not None:
        #            add_summary_value(tf_summary_writer, 'train_loss', loss, iteration)
        #            add_summary_value(tf_summary_writer, 'learning_rate', opt.learning_rate, iteration)
        #            # add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
        #            if opt.self_critical:
        #                add_summary_value(tf_summary_writer, 'cider_score', cider_score.data.item(), iteration)
        #        
        #            tf_summary_writer.flush()


        #ss_prob_history[iteration] = model.ss_prob

def eval(opt):
    model.eval()
    #########################################################################################
    # eval begins here
    #########################################################################################
    data_iter_val = iter(dataloader_val)
    #loss_temp = 0
    #start = time.time()

    num_show = 0
    predictions = []
    count = 0
    for step in range(len(dataloader_val)):
        data = data_iter_val.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id = data

        proposals = proposals[:,:max(int(max(num[:,1])),1),:]

        input_imgs.data.resize_(img.size()).copy_(img)
        input_seqs.data.resize_(iseq.size()).copy_(iseq)
        gt_seqs.data.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.data.resize_(num.size()).copy_(num)
        input_ppls.data.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.data.resize_(bboxs.size()).copy_(bboxs)
        mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)
        input_imgs.data.resize_(img.size()).copy_(img)

        eval_opt = {'sample_max':1, 'beam_size': opt.beam_size, 'inference_mode' : True, 'tag_size' : opt.cbs_tag_size}
        seq, bn_seq, fg_seq =  model(input_imgs, input_seqs, gt_seqs, input_num, input_ppls, gt_bboxs, mask_bboxs, 'sample', eval_opt)
        sents = utils.decode_sequence(dataset.itow, dataset.itod, dataset.ltow, dataset.itoc, dataset.wtod, seq.data, bn_seq.data, fg_seq.data, opt.vocab_size, opt)
        for k, sent in enumerate(sents):
            entry = {'image_id': img_id[k].item(), 'caption': sent}
            predictions.append(entry)
            
            if num_show < opt.batch_size:
                print('image %s: %s' % (entry['image_id'], entry['caption']) )
                num_show += 1

        if count % 100 == 0:
            print("Magi_ZZ_ML_Kernel:>> Evaluation function just ran for %d times..."%count)
        count += 1

    print('Magi_ZZ_ML_Kernel:>> Total images and captions to be evaluated is: %d' %(len(predictions)))
    lang_stats = None
    if opt.language_eval == 1:
        #if opt.decode_noc:
            #lang_stats = utils.noc_eval(predictions, str(1), opt.val_split, opt)
        #else:
        lang_stats = utils.language_eval(opt.dataset, predictions, str(1), opt.val_split, opt)


    print('Magi_ZZ_ML_Kernel:>> Saving the predictions...')
    if opt.inference_only:
        opt.beam_size=3
        lang_stats = utils.language_eval(opt.dataset, predictions, str(1), opt.val_split, opt)
        print("Magi_ZZ_ML_Kernel:>> Welcome To Inference mode, saving scores into {} ", opt.checkpoint_path)
        with open(os.path.join(opt.checkpoint_path, 'lang_stats.json'), 'w') as f:
            json.dump(lang_stats, f)
        print("Magi_ZZ_ML_Kernel:>> Done!")
        print("Magi_ZZ_ML_Kernel:>> now saving images and captions into {} ", opt.checkpoint_path)
        with open(os.path.join(opt.checkpoint_path, 'preds.json'), 'w') as f:
            json.dump(predictions, f)
        print("Magi_ZZ_ML_Kernel:>> Done!")
        print("Magi_ZZ_ML_Kernel:>> now saving images and captions into {} ", opt.checkpoint_path)
        with open(os.path.join(opt.checkpoint_path, 'sents.json'), 'w') as f:
            json.dump(sents, f)
        print("Magi_ZZ_ML_Kernel:>> Done!")

    # Write validation result into summary
    if tf is not None and opt.val_split=='val':
        for k,v in lang_stats.items():
            add_summary_value(tf_summary_writer, k, v, iteration)
        tf_summary_writer.flush()


    return lang_stats


####################################################################################
# Main
####################################################################################
# initialize the data holder.

if __name__ == "__main__":
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        utils.update_values(options_yaml, vars(opt))
    
    print("Magi_ZZ_ML_Kernel:>> Welcome to neural baby talk by jiasen lu et al. revived and repaired by Zanyar.Z.\n")
    print("Magi_ZZ_ML_Kernel:>> Printing the whole operation and it's initial parameters...\n")
    input("Press any key to continue...")
    if opt.inference_only:
        print("Inference Mode (Caption Generation Mode) Detected...\n")
        input("Press any key to continue...")
    else:
        print("Training Mode Detected...\n")
        input("Press any key to continue...")

    print(opt)
    cudnn.benchmark = True

    if opt.dataset == 'flickr30k':
        from misc.dataloader_flickr30k import DataLoader
    else:
        from misc.dataloader_coco import DataLoader

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    ####################################################################################
    # Data Loader
    ####################################################################################
    dataset = DataLoader(opt, split='train')
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,shuffle=False, num_workers=opt.num_workers)

    dataset_val = DataLoader(opt, split=opt.val_split)
    
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,shuffle=False, num_workers=opt.num_workers)
    
    input_imgs = torch.FloatTensor(1)
    input_seqs = torch.LongTensor(1)
    input_ppls = torch.FloatTensor(1)
    gt_bboxs = torch.FloatTensor(1)
    mask_bboxs = torch.ByteTensor(1)
    gt_seqs = torch.LongTensor(1)
    input_num = torch.LongTensor(1)

    
    input_imgs = input_imgs.cuda()
    input_seqs = input_seqs.cuda()
    gt_seqs = gt_seqs.cuda()
    input_num = input_num.cuda()
    input_ppls = input_ppls.cuda()
    gt_bboxs = gt_bboxs.cuda()
    mask_bboxs = mask_bboxs.cuda()

    input_imgs = Variable(input_imgs)
    input_seqs = Variable(input_seqs)
    gt_seqs = Variable(gt_seqs)
    input_num = Variable(input_num)
    input_ppls = Variable(input_ppls)
    gt_bboxs = Variable(gt_bboxs)
    mask_bboxs = Variable(mask_bboxs)

    ####################################################################################
    # Build the Model
    ####################################################################################
    opt.vocab_size = dataset.vocab_size
    opt.detect_size = dataset.detect_size
    opt.seq_length = opt.seq_length
    opt.fg_size = dataset.fg_size
    opt.fg_mask = torch.from_numpy(dataset.fg_mask).byte()
    opt.st2towidx = torch.from_numpy(dataset.st2towidx).long()


    if opt.bert_base_768:
        opt.glove_fg = torch.from_numpy(dataset.bert_fg).float()
        opt.glove_clss = torch.from_numpy(dataset.bert_clss).float()
        opt.glove_w = torch.from_numpy(dataset.bert_w).float()


    if opt.glove_6B_300:
        opt.glove_fg = torch.from_numpy(dataset.glove_fg).float()
        opt.glove_clss = torch.from_numpy(dataset.glove_clss).float()
        opt.glove_w = torch.from_numpy(dataset.glove_w).float()
        

    opt.itow = dataset.itow
    opt.itod = dataset.itod
    opt.ltow = dataset.ltow
    opt.itoc = dataset.itoc

    if not opt.finetune_cnn: opt.fixed_block = 4 # if not finetune, fix all cnn block

    if opt.att_model == 'topdown':
        model = AttModel.TopDownModel(opt)
    elif opt.att_model == 'att2in2':
        model = AttModel.Att2in2Model(opt)
    elif opt.att_model == 'newtopdown':
        model = AttModel.NewTopDownModel(opt)
    
    

    tf_summary_writer = tf and tf.Summary.FileWriter(opt.checkpoint_path)
    start_epoch=0
    best_val_score=None
    # infos = {}
    # histories = {}
    if opt.start_from is not None:
        if opt.load_best_score == 1:
            model_path = os.path.join(opt.start_from, 'model-best.pth')
            # info_path = os.path.join(opt.start_from, 'infos_'+opt.id+'-best.pkl')
        else:
            model_path = os.path.join(opt.start_from, 'model.pth')
            # info_path = os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')

            # open old infos and check if models are compatible
        
        # with open(info_path, 'rb') as f:
        #     infos = pickle.load(f,encoding='latin1')
        #     saved_model_opt = infos['opt']

        # opt.learning_rate = saved_model_opt.learning_rate
        print('Magi_ZZ_ML_Kernel:>> Loading the model %s...' %(model_path))
        checkpoint=torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_score = checkpoint['best_val_score']
        print('current:{}'.format(checkpoint['info']))

    if opt.decode_noc and opt.start_from is None:
        model._reinit_word_weight(opt, dataset.ctoi, dataset.wtoi)

    # iteration = infos.get('iter', 0)
    iteration=0

    if opt.mGPUs:
        model = nn.DataParallel(model)

    if opt.cuda:
        model.cuda()

    params = []
    # cnn_params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'cnn' in key:
                params += [{'params':[value], 'lr':opt.cnn_learning_rate,
                        'weight_decay':opt.cnn_weight_decay, 'betas':(opt.cnn_optim_alpha, opt.cnn_optim_beta)}]
            else:
                params += [{'params':[value], 'lr':opt.learning_rate,
                    'weight_decay':opt.weight_decay, 'betas':(opt.optim_alpha, opt.optim_beta)}]

    print("Magi_ZZ_ML_Kernel:>> Using %s as our optimization method..." %(opt.optim))
   
    if opt.optim == 'sgd':
        optimizer = optim.SGD(params, momentum=0.9)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(params)
    elif opt.optim == 'adamax':
        optimizer = optim.Adamax(params)
        
    # if opt.start_from is not None:
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, opt.max_epochs):
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
            if (epoch - opt.learning_rate_decay_start) % opt.learning_rate_decay_every == 0:
                # decay the learning rate.
                utils.set_lr(optimizer, opt.learning_rate_decay_rate)
                opt.learning_rate  = opt.learning_rate * opt.learning_rate_decay_rate

        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
            # if opt.start_from is None or epoch==start_epoch:
            model.ss_prob = opt.ss_prob

        if not opt.inference_only:
            train(epoch, opt)

        if epoch % opt.val_every_epoch == 0:
            
            lang_stats = eval(opt)

            # Save model if is improving on validation result
            current_score = lang_stats['CIDEr']

            best_flag = False

            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            if opt.mGPUs:
                state = {'net':model.module.state_dict(), 'optimizer':optimizer.state_dict(),\
                    'epoch':epoch,'info': lang_stats,'best_val_score':best_val_score}
            else:
                state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(),\
                    'epoch':epoch,'info': lang_stats,'best_val_score':best_val_score}

            torch.save(state, checkpoint_path)
            
            print("Magi_ZZ_ML_Kernel:>> model saved to {}".format(checkpoint_path))

            # Dump miscalleous informations
            # infos['epoch'] = epoch
            # infos['best_val_score'] = best_val_score
            # infos['opt'] = opt
            # infos['all'] = lang_stats
            # infos['vocab'] = dataset.itow


            with open(os.path.join(opt.checkpoint_path, 'myhist.json'), 'a') as f:
                json.dump(lang_stats, f)
            # with open(os.path.join(opt.checkpoint_path, 'infos_'+str(epoch)+'.pkl'), 'wb') as f:
            #     pickle.dump(infos, f)


            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                torch.save(state, checkpoint_path)

                print("Magi_ZZ_ML_Kernel:>> model saved to {} with best cider score {:.3f}".format(checkpoint_path, best_val_score))
                # with open(os.path.join(opt.checkpoint_path, 'infos_'+str(epoch)+'-best.pkl'), 'wb') as f:
                #     pickle.dump(infos, f)

    # 最终测试
    opt.inference_only=True
    opt.val_split='test'
    print('final test..')
    dataset_val = DataLoader(opt, split='test')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,shuffle=False, num_workers=opt.num_workers)
    lang_stats = eval(opt)
    with open(os.path.join(opt.checkpoint_path, 'final_test.json'), 'a') as f:
                json.dump(lang_stats, f)