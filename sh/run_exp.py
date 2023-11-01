#!/usr/bin/env python
import os
import argparse

def get_cmd(task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch, warmup,
            model_dir, summary_dir, res_fn, load_model_dir, tag_suffix):
    cmd_str = 'bash exp_with_args.sh %s %s %s %d %d %d %d %d %d %d %d %d %s %s %s %s %s' % \
              (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
               warmup, model_dir, summary_dir, res_fn, load_model_dir, tag_suffix)
    return cmd_str

def get_cmd_meta_train_crossfit(task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch, warmup,
            model_dir, summary_dir, res_fn, inner_lr, task_bs, logging, meta_algo):
    cmd_str = 'bash exp_with_args_meta_train.sh %s %s %s %d %d %d %d %d %d %d %d %d %s %s %s %d %d %d %s' % \
              (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
               warmup, model_dir, summary_dir, res_fn, inner_lr, task_bs, logging, meta_algo)
    return cmd_str

def get_args_by_task_model(task, sub_task, model_tag):
    if task in ['tfix_high_resource',
                'tfix_low_resource_full',
                'tfix_high_resource_meta_crossfit',
                'tfix_low_resource_100_shot_100_seed',
                'tfix_low_resource_100_shot_13_seed',
                'tfix_low_resource_100_shot_21_seed',
                'tfix_low_resource_100_shot_42_seed',
                'tfix_low_resource_100_shot_87_seed',
                'tfix_low_resource_50_shot_100_seed',
                'tfix_low_resource_50_shot_13_seed',
                'tfix_low_resource_50_shot_21_seed',
                'tfix_low_resource_50_shot_42_seed',
                'tfix_low_resource_50_shot_87_seed',
                'tfix_low_resource_10_shot_100_seed',
                'tfix_low_resource_10_shot_13_seed',
                'tfix_low_resource_10_shot_21_seed',
                'tfix_low_resource_10_shot_42_seed',
                'tfix_low_resource_10_shot_87_seed',
                'tfix_joint_100_shot',
                'tfix_joint_50_shot',
                'tfix_joint_10_shot',
                'joint_upsampling_tfix_low_resource_100_shot_100_seed',
                'joint_upsampling_tfix_low_resource_100_shot_13_seed',
                'joint_upsampling_tfix_low_resource_100_shot_21_seed',
                'joint_upsampling_tfix_low_resource_100_shot_42_seed',
                'joint_upsampling_tfix_low_resource_100_shot_87_seed',
                'joint_upsampling_tfix_low_resource_50_shot_100_seed',
                'joint_upsampling_tfix_low_resource_50_shot_13_seed',
                'joint_upsampling_tfix_low_resource_50_shot_21_seed',
                'joint_upsampling_tfix_low_resource_50_shot_42_seed',
                'joint_upsampling_tfix_low_resource_50_shot_87_seed',
                'joint_upsampling_tfix_low_resource_10_shot_100_seed',
                'joint_upsampling_tfix_low_resource_10_shot_13_seed',
                'joint_upsampling_tfix_low_resource_10_shot_21_seed',
                'joint_upsampling_tfix_low_resource_10_shot_42_seed',
                'joint_upsampling_tfix_low_resource_10_shot_87_seed']:
        src_len = 400
        trg_len = 256
        epoch = 50
        patience = 50
    elif task in ['sb4j_high_resource',
                  'sb4j_low_resource_full',
                  'sb4j_high_resource_meta_crossfit',
                  'sb4j_low_resource_100_shot_100_seed',
                  'sb4j_low_resource_100_shot_13_seed',
                  'sb4j_low_resource_100_shot_21_seed',
                  'sb4j_low_resource_100_shot_42_seed',
                  'sb4j_low_resource_100_shot_87_seed',
                  'sb4j_low_resource_50_shot_100_seed',
                  'sb4j_low_resource_50_shot_13_seed',
                  'sb4j_low_resource_50_shot_21_seed',
                  'sb4j_low_resource_50_shot_42_seed',
                  'sb4j_low_resource_50_shot_87_seed',
                  'sb4j_low_resource_10_shot_100_seed',
                  'sb4j_low_resource_10_shot_13_seed',
                  'sb4j_low_resource_10_shot_21_seed',
                  'sb4j_low_resource_10_shot_42_seed',
                  'sb4j_low_resource_10_shot_87_seed',
                  'sb4j_joint_100_shot',
                  'sb4j_joint_50_shot',
                  'sb4j_joint_10_shot',
                  'joint_upsampling_sb4j_low_resource_100_shot_100_seed',
                  'joint_upsampling_sb4j_low_resource_100_shot_13_seed',
                  'joint_upsampling_sb4j_low_resource_100_shot_21_seed',
                  'joint_upsampling_sb4j_low_resource_100_shot_42_seed',
                  'joint_upsampling_sb4j_low_resource_100_shot_87_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_100_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_13_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_21_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_42_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_87_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_100_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_13_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_21_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_42_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_87_seed']:
        src_len = 256
        trg_len = 256
        epoch = 50
        patience = 50
    elif task in ['tssb_high_resource',
                  'tssb_low_resource_full',
                  'tssb_high_resource_meta_crossfit',
                  'tssb_low_resource_100_shot_100_seed',
                  'tssb_low_resource_100_shot_13_seed',
                  'tssb_low_resource_100_shot_21_seed',
                  'tssb_low_resource_100_shot_42_seed',
                  'tssb_low_resource_100_shot_87_seed',
                  'tssb_low_resource_50_shot_100_seed',
                  'tssb_low_resource_50_shot_13_seed',
                  'tssb_low_resource_50_shot_21_seed',
                  'tssb_low_resource_50_shot_42_seed',
                  'tssb_low_resource_50_shot_87_seed',
                  'tssb_low_resource_10_shot_100_seed',
                  'tssb_low_resource_10_shot_13_seed',
                  'tssb_low_resource_10_shot_21_seed',
                  'tssb_low_resource_10_shot_42_seed',
                  'tssb_low_resource_10_shot_87_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_100_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_13_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_21_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_42_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_87_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_100_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_13_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_21_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_42_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_87_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_100_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_13_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_21_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_42_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_87_seed',
                  'tssb_joint_100_shot',
                  'tssb_joint_50_shot',
                  'tssb_joint_10_shot']:
        src_len = 256
        trg_len = 256
        epoch = 50
        patience = 50

    if 'codet5_small' in model_tag:
        bs = 64
        if task == 'tfix_high_resource_meta_crossfit':
            bs = 10
        elif task == 'sb4j_high_resource_meta_crossfit':
            bs = 10
        elif task == 'tssb_high_resource_meta_crossfit':
            bs = 10
    else:
        bs = 25
        if task == 'tfix_high_resource_meta_crossfit':
            bs = 10
        elif task == 'sb4j_high_resource_meta_crossfit':
            bs = 10
        elif task == 'tssb_high_resource_meta_crossfit':
            bs = 10

    # lr = 5
    if 'tfix' in task:
        lr = 10
    else:
        lr = 5

    return bs, lr, src_len, trg_len, patience, epoch


def run_one_exp(args):
    bs, lr, src_len, trg_len, patience, epoch = get_args_by_task_model(args.task, args.sub_task, args.model_tag)
    if args.do_meta_train_crossfit:
        print('============================Start Meta Training Crossfit==========================')
        if args.do_reptile:
            algo = 'reptile'
        elif args.do_maml:
            algo = 'maml'

        cmd_str = get_cmd_meta_train_crossfit(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag,
                                              gpu=args.gpu,
                                              data_num=args.data_num, bs=bs, lr=lr, source_length=src_len,
                                              target_length=trg_len,
                                              patience=patience, epoch=epoch, warmup=1000,
                                              model_dir=args.model_dir, summary_dir=args.summary_dir,
                                              res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag),
                                              inner_lr=args.inner_learning_rate,
                                              task_bs=args.task_batch_size, logging=10, meta_algo=algo)
    else:
        print('============================Start Running==========================')
        cmd_str = get_cmd(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag, gpu=args.gpu,
                          data_num=args.data_num, bs=bs, lr=lr, source_length=src_len, target_length=trg_len,
                          patience=patience, epoch=epoch, warmup=1000,
                          model_dir=args.model_dir, summary_dir=args.summary_dir,
                          res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag),
                          load_model_dir=args.load_model_dir, tag_suffix=args.tag_suffix)
    print('%s\n' % cmd_str)
    os.system(cmd_str)


def get_sub_tasks(task):
    if task in ['refine']:
        sub_tasks = ['small', 'medium']
    elif task in ['tfix_high_resource',
                  'tfix_low_resource_full',
                  'tfix_high_resource_meta_crossfit',
                  'tfix_low_resource_100_shot_100_seed',
                  'tfix_low_resource_100_shot_13_seed',
                  'tfix_low_resource_100_shot_21_seed',
                  'tfix_low_resource_100_shot_42_seed',
                  'tfix_low_resource_100_shot_87_seed',
                  'tfix_low_resource_50_shot_100_seed',
                  'tfix_low_resource_50_shot_13_seed',
                  'tfix_low_resource_50_shot_21_seed',
                  'tfix_low_resource_50_shot_42_seed',
                  'tfix_low_resource_50_shot_87_seed',
                  'tfix_low_resource_10_shot_100_seed',
                  'tfix_low_resource_10_shot_13_seed',
                  'tfix_low_resource_10_shot_21_seed',
                  'tfix_low_resource_10_shot_42_seed',
                  'tfix_low_resource_10_shot_87_seed',
                  'tfix_joint_100_shot',
                  'tfix_joint_50_shot',
                  'tfix_joint_10_shot',
                  'joint_upsampling_tfix_low_resource_100_shot_100_seed',
                  'joint_upsampling_tfix_low_resource_100_shot_13_seed',
                  'joint_upsampling_tfix_low_resource_100_shot_21_seed',
                  'joint_upsampling_tfix_low_resource_100_shot_42_seed',
                  'joint_upsampling_tfix_low_resource_100_shot_87_seed',
                  'joint_upsampling_tfix_low_resource_50_shot_100_seed',
                  'joint_upsampling_tfix_low_resource_50_shot_13_seed',
                  'joint_upsampling_tfix_low_resource_50_shot_21_seed',
                  'joint_upsampling_tfix_low_resource_50_shot_42_seed',
                  'joint_upsampling_tfix_low_resource_50_shot_87_seed',
                  'joint_upsampling_tfix_low_resource_10_shot_100_seed',
                  'joint_upsampling_tfix_low_resource_10_shot_13_seed',
                  'joint_upsampling_tfix_low_resource_10_shot_21_seed',
                  'joint_upsampling_tfix_low_resource_10_shot_42_seed',
                  'joint_upsampling_tfix_low_resource_10_shot_87_seed',
                  'sb4j_high_resource',
                  'sb4j_low_resource_full',
                  'sb4j_high_resource_meta_crossfit',
                  'sb4j_low_resource_100_shot_100_seed',
                  'sb4j_low_resource_100_shot_13_seed',
                  'sb4j_low_resource_100_shot_21_seed',
                  'sb4j_low_resource_100_shot_42_seed',
                  'sb4j_low_resource_100_shot_87_seed',
                  'sb4j_low_resource_50_shot_100_seed',
                  'sb4j_low_resource_50_shot_13_seed',
                  'sb4j_low_resource_50_shot_21_seed',
                  'sb4j_low_resource_50_shot_42_seed',
                  'sb4j_low_resource_50_shot_87_seed',
                  'sb4j_low_resource_10_shot_100_seed',
                  'sb4j_low_resource_10_shot_13_seed',
                  'sb4j_low_resource_10_shot_21_seed',
                  'sb4j_low_resource_10_shot_42_seed',
                  'sb4j_low_resource_10_shot_87_seed',
                  'sb4j_joint_100_shot',
                  'sb4j_joint_50_shot',
                  'sb4j_joint_10_shot',
                  'joint_upsampling_sb4j_low_resource_100_shot_100_seed',
                  'joint_upsampling_sb4j_low_resource_100_shot_13_seed',
                  'joint_upsampling_sb4j_low_resource_100_shot_21_seed',
                  'joint_upsampling_sb4j_low_resource_100_shot_42_seed',
                  'joint_upsampling_sb4j_low_resource_100_shot_87_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_100_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_13_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_21_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_42_seed',
                  'joint_upsampling_sb4j_low_resource_50_shot_87_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_100_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_13_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_21_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_42_seed',
                  'joint_upsampling_sb4j_low_resource_10_shot_87_seed',
                  'tssb_high_resource',
                  'tssb_low_resource_full',
                  'tssb_high_resource_meta_crossfit',
                  'tssb_low_resource_100_shot_100_seed',
                  'tssb_low_resource_100_shot_13_seed',
                  'tssb_low_resource_100_shot_21_seed',
                  'tssb_low_resource_100_shot_42_seed',
                  'tssb_low_resource_100_shot_87_seed',
                  'tssb_low_resource_50_shot_100_seed',
                  'tssb_low_resource_50_shot_13_seed',
                  'tssb_low_resource_50_shot_21_seed',
                  'tssb_low_resource_50_shot_42_seed',
                  'tssb_low_resource_50_shot_87_seed',
                  'tssb_low_resource_10_shot_100_seed',
                  'tssb_low_resource_10_shot_13_seed',
                  'tssb_low_resource_10_shot_21_seed',
                  'tssb_low_resource_10_shot_42_seed',
                  'tssb_low_resource_10_shot_87_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_100_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_13_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_21_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_42_seed',
                  'joint_upsampling_tssb_low_resource_100_shot_87_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_100_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_13_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_21_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_42_seed',
                  'joint_upsampling_tssb_low_resource_50_shot_87_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_100_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_13_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_21_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_42_seed',
                  'joint_upsampling_tssb_low_resource_10_shot_87_seed',
                  'tssb_joint_100_shot',
                  'tssb_joint_50_shot',
                  'tssb_joint_10_shot']:
        sub_tasks = ['none']
    return sub_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='codet5_small',
                        choices=['roberta', 'codebert', 'bart_base', 't5_base', 'codet5_small', 'codet5_base',
                                 'unixcoder_base'])
    parser.add_argument("--task", type=str, default='tfix_low_resource_full',
                        choices=['tfix_high_resource',
                                 'tfix_low_resource_full',
                                 'tfix_high_resource_meta_crossfit',
                                 'tfix_low_resource_100_shot_100_seed',
                                 'tfix_low_resource_100_shot_13_seed',
                                 'tfix_low_resource_100_shot_21_seed',
                                 'tfix_low_resource_100_shot_42_seed',
                                 'tfix_low_resource_100_shot_87_seed',
                                 'tfix_low_resource_50_shot_100_seed',
                                 'tfix_low_resource_50_shot_13_seed',
                                 'tfix_low_resource_50_shot_21_seed',
                                 'tfix_low_resource_50_shot_42_seed',
                                 'tfix_low_resource_50_shot_87_seed',
                                 'tfix_low_resource_10_shot_100_seed',
                                 'tfix_low_resource_10_shot_13_seed',
                                 'tfix_low_resource_10_shot_21_seed',
                                 'tfix_low_resource_10_shot_42_seed',
                                 'tfix_low_resource_10_shot_87_seed',
                                 'tfix_joint_100_shot',
                                 'tfix_joint_50_shot',
                                 'tfix_joint_10_shot',
                                 'joint_upsampling_tfix_low_resource_100_shot_100_seed',
                                 'joint_upsampling_tfix_low_resource_100_shot_13_seed',
                                 'joint_upsampling_tfix_low_resource_100_shot_21_seed',
                                 'joint_upsampling_tfix_low_resource_100_shot_42_seed',
                                 'joint_upsampling_tfix_low_resource_100_shot_87_seed',
                                 'joint_upsampling_tfix_low_resource_50_shot_100_seed',
                                 'joint_upsampling_tfix_low_resource_50_shot_13_seed',
                                 'joint_upsampling_tfix_low_resource_50_shot_21_seed',
                                 'joint_upsampling_tfix_low_resource_50_shot_42_seed',
                                 'joint_upsampling_tfix_low_resource_50_shot_87_seed',
                                 'joint_upsampling_tfix_low_resource_10_shot_100_seed',
                                 'joint_upsampling_tfix_low_resource_10_shot_13_seed',
                                 'joint_upsampling_tfix_low_resource_10_shot_21_seed',
                                 'joint_upsampling_tfix_low_resource_10_shot_42_seed',
                                 'joint_upsampling_tfix_low_resource_10_shot_87_seed',
                                 'sb4j_high_resource',
                                 'sb4j_low_resource_full',
                                 'sb4j_high_resource_meta_crossfit',
                                 'sb4j_low_resource_100_shot_100_seed',
                                 'sb4j_low_resource_100_shot_13_seed',
                                 'sb4j_low_resource_100_shot_21_seed',
                                 'sb4j_low_resource_100_shot_42_seed',
                                 'sb4j_low_resource_100_shot_87_seed',
                                 'sb4j_low_resource_50_shot_100_seed',
                                 'sb4j_low_resource_50_shot_13_seed',
                                 'sb4j_low_resource_50_shot_21_seed',
                                 'sb4j_low_resource_50_shot_42_seed',
                                 'sb4j_low_resource_50_shot_87_seed',
                                 'sb4j_low_resource_10_shot_100_seed',
                                 'sb4j_low_resource_10_shot_13_seed',
                                 'sb4j_low_resource_10_shot_21_seed',
                                 'sb4j_low_resource_10_shot_42_seed',
                                 'sb4j_low_resource_10_shot_87_seed',
                                 'sb4j_joint_100_shot',
                                 'sb4j_joint_50_shot',
                                 'sb4j_joint_10_shot',
                                 'joint_upsampling_sb4j_low_resource_100_shot_100_seed',
                                 'joint_upsampling_sb4j_low_resource_100_shot_13_seed',
                                 'joint_upsampling_sb4j_low_resource_100_shot_21_seed',
                                 'joint_upsampling_sb4j_low_resource_100_shot_42_seed',
                                 'joint_upsampling_sb4j_low_resource_100_shot_87_seed',
                                 'joint_upsampling_sb4j_low_resource_50_shot_100_seed',
                                 'joint_upsampling_sb4j_low_resource_50_shot_13_seed',
                                 'joint_upsampling_sb4j_low_resource_50_shot_21_seed',
                                 'joint_upsampling_sb4j_low_resource_50_shot_42_seed',
                                 'joint_upsampling_sb4j_low_resource_50_shot_87_seed',
                                 'joint_upsampling_sb4j_low_resource_10_shot_100_seed',
                                 'joint_upsampling_sb4j_low_resource_10_shot_13_seed',
                                 'joint_upsampling_sb4j_low_resource_10_shot_21_seed',
                                 'joint_upsampling_sb4j_low_resource_10_shot_42_seed',
                                 'joint_upsampling_sb4j_low_resource_10_shot_87_seed',
                                 'tssb_high_resource',
                                 'tssb_low_resource_full',
                                 'tssb_high_resource_meta_crossfit',
                                 'tssb_low_resource_100_shot_100_seed',
                                 'tssb_low_resource_100_shot_13_seed',
                                 'tssb_low_resource_100_shot_21_seed',
                                 'tssb_low_resource_100_shot_42_seed',
                                 'tssb_low_resource_100_shot_87_seed',
                                 'tssb_low_resource_50_shot_100_seed',
                                 'tssb_low_resource_50_shot_13_seed',
                                 'tssb_low_resource_50_shot_21_seed',
                                 'tssb_low_resource_50_shot_42_seed',
                                 'tssb_low_resource_50_shot_87_seed',
                                 'tssb_low_resource_10_shot_100_seed',
                                 'tssb_low_resource_10_shot_13_seed',
                                 'tssb_low_resource_10_shot_21_seed',
                                 'tssb_low_resource_10_shot_42_seed',
                                 'tssb_low_resource_10_shot_87_seed',
                                 'joint_upsampling_tssb_low_resource_100_shot_100_seed',
                                 'joint_upsampling_tssb_low_resource_100_shot_13_seed',
                                 'joint_upsampling_tssb_low_resource_100_shot_21_seed',
                                 'joint_upsampling_tssb_low_resource_100_shot_42_seed',
                                 'joint_upsampling_tssb_low_resource_100_shot_87_seed',
                                 'joint_upsampling_tssb_low_resource_50_shot_100_seed',
                                 'joint_upsampling_tssb_low_resource_50_shot_13_seed',
                                 'joint_upsampling_tssb_low_resource_50_shot_21_seed',
                                 'joint_upsampling_tssb_low_resource_50_shot_42_seed',
                                 'joint_upsampling_tssb_low_resource_50_shot_87_seed',
                                 'joint_upsampling_tssb_low_resource_10_shot_100_seed',
                                 'joint_upsampling_tssb_low_resource_10_shot_13_seed',
                                 'joint_upsampling_tssb_low_resource_10_shot_21_seed',
                                 'joint_upsampling_tssb_low_resource_10_shot_42_seed',
                                 'joint_upsampling_tssb_low_resource_10_shot_87_seed',
                                 'tssb_joint_100_shot',
                                 'tssb_joint_50_shot',
                                 'tssb_joint_10_shot'])
    parser.add_argument("--sub_task", type=str, default='none')
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=int, default=0, help='index of the gpu to use in a cluster')
    parser.add_argument("--do_meta_train_crossfit", action='store_true',
                        help="Whether to run white box meta train by following crossfit.")
    parser.add_argument("--load_model_dir", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--tag_suffix", default='finetune', type=str,
                        help="Experiment full model tag suffix")
    parser.add_argument("--do_maml", action='store_true',
                        help="Whether to run maml.")
    parser.add_argument("--do_reptile", action='store_true',
                        help="Whether to run reptile.")
    parser.add_argument("--inner_learning_rate", default=10, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--task_batch_size", default=150, type=int,
                        help="Task batch size for MAML.")

    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    assert args.sub_task in get_sub_tasks(args.task)
    run_one_exp(args)
