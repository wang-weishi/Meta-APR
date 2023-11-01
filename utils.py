import pdb
from torch.nn.init import xavier_uniform_
from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from _utils import *

logger = logging.getLogger(__name__)

def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    if args.do_maml and split_tag == 'train':
        cache_fn = '{}/{}.pt'.format(args.cache_path,
                                     args.sub_task + '-' + split_tag + ('_src' if only_src else '') + data_tag)
    else:
        cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    examples = read_examples(filename, args.data_num, args.task)

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if split_tag == 'train' or split_tag == 'test':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10k data for evaluation from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data

def load_and_cache_clone_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + '_all' if args.data_num == -1 else '_%d' % args.data_num)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_clone_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def get_filenames(data_root, task, sub_task, split=''):
    if 'tfix' in task:
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.pkl'.format(data_dir)
        dev_fn = '{}/val.pkl'.format(data_dir)
        test_fn = '{}/test.pkl'.format(data_dir)
    elif 'sb4j' in task:
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.pkl'.format(data_dir)
        dev_fn = '{}/val.pkl'.format(data_dir)
        test_fn = '{}/test.pkl'.format(data_dir)
    elif 'tssb' in task:
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.pkl'.format(data_dir)
        dev_fn = '{}/val.pkl'.format(data_dir)
        test_fn = '{}/test.pkl'.format(data_dir)


    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'tfix_high_resource': read_tfix_examples,
        'tfix_low_resource_full': read_tfix_examples,
        'tfix_high_resource_meta_crossfit': read_tfix_examples,
        'tfix_low_resource_100_shot_100_seed': read_tfix_examples,
        'tfix_low_resource_100_shot_13_seed': read_tfix_examples,
        'tfix_low_resource_100_shot_21_seed': read_tfix_examples,
        'tfix_low_resource_100_shot_42_seed': read_tfix_examples,
        'tfix_low_resource_100_shot_87_seed': read_tfix_examples,
        'tfix_low_resource_50_shot_100_seed': read_tfix_examples,
        'tfix_low_resource_50_shot_13_seed': read_tfix_examples,
        'tfix_low_resource_50_shot_21_seed': read_tfix_examples,
        'tfix_low_resource_50_shot_42_seed': read_tfix_examples,
        'tfix_low_resource_50_shot_87_seed': read_tfix_examples,
        'tfix_low_resource_10_shot_100_seed': read_tfix_examples,
        'tfix_low_resource_10_shot_13_seed': read_tfix_examples,
        'tfix_low_resource_10_shot_21_seed': read_tfix_examples,
        'tfix_low_resource_10_shot_42_seed': read_tfix_examples,
        'tfix_low_resource_10_shot_87_seed': read_tfix_examples,
        'tfix_joint_100_shot': read_tfix_examples,
        'tfix_joint_50_shot': read_tfix_examples,
        'tfix_joint_10_shot': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_100_shot_100_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_100_shot_13_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_100_shot_21_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_100_shot_42_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_100_shot_87_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_50_shot_100_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_50_shot_13_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_50_shot_21_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_50_shot_42_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_50_shot_87_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_10_shot_100_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_10_shot_13_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_10_shot_21_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_10_shot_42_seed': read_tfix_examples,
        'joint_upsampling_tfix_low_resource_10_shot_87_seed': read_tfix_examples,
        'sb4j_high_resource': read_tfix_examples,
        'sb4j_low_resource_full': read_tfix_examples,
        'sb4j_high_resource_meta_crossfit': read_tfix_examples,
        'sb4j_low_resource_100_shot_100_seed': read_tfix_examples,
        'sb4j_low_resource_100_shot_13_seed': read_tfix_examples,
        'sb4j_low_resource_100_shot_21_seed': read_tfix_examples,
        'sb4j_low_resource_100_shot_42_seed': read_tfix_examples,
        'sb4j_low_resource_100_shot_87_seed': read_tfix_examples,
        'sb4j_low_resource_50_shot_100_seed': read_tfix_examples,
        'sb4j_low_resource_50_shot_13_seed': read_tfix_examples,
        'sb4j_low_resource_50_shot_21_seed': read_tfix_examples,
        'sb4j_low_resource_50_shot_42_seed': read_tfix_examples,
        'sb4j_low_resource_50_shot_87_seed': read_tfix_examples,
        'sb4j_low_resource_10_shot_100_seed': read_tfix_examples,
        'sb4j_low_resource_10_shot_13_seed': read_tfix_examples,
        'sb4j_low_resource_10_shot_21_seed': read_tfix_examples,
        'sb4j_low_resource_10_shot_42_seed': read_tfix_examples,
        'sb4j_low_resource_10_shot_87_seed': read_tfix_examples,
        'sb4j_joint_100_shot': read_tfix_examples,
        'sb4j_joint_50_shot': read_tfix_examples,
        'sb4j_joint_10_shot': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_100_shot_100_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_100_shot_13_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_100_shot_21_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_100_shot_42_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_100_shot_87_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_50_shot_100_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_50_shot_13_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_50_shot_21_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_50_shot_42_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_50_shot_87_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_10_shot_100_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_10_shot_13_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_10_shot_21_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_10_shot_42_seed': read_tfix_examples,
        'joint_upsampling_sb4j_low_resource_10_shot_87_seed': read_tfix_examples,
        'tssb_high_resource': read_tfix_examples,
        'tssb_low_resource_full': read_tfix_examples,
        'tssb_high_resource_meta_crossfit': read_tfix_examples,
        'tssb_low_resource_100_shot_100_seed': read_tfix_examples,
        'tssb_low_resource_100_shot_13_seed': read_tfix_examples,
        'tssb_low_resource_100_shot_21_seed': read_tfix_examples,
        'tssb_low_resource_100_shot_42_seed': read_tfix_examples,
        'tssb_low_resource_100_shot_87_seed': read_tfix_examples,
        'tssb_low_resource_50_shot_100_seed': read_tfix_examples,
        'tssb_low_resource_50_shot_13_seed': read_tfix_examples,
        'tssb_low_resource_50_shot_21_seed': read_tfix_examples,
        'tssb_low_resource_50_shot_42_seed': read_tfix_examples,
        'tssb_low_resource_50_shot_87_seed': read_tfix_examples,
        'tssb_low_resource_10_shot_100_seed': read_tfix_examples,
        'tssb_low_resource_10_shot_13_seed': read_tfix_examples,
        'tssb_low_resource_10_shot_21_seed': read_tfix_examples,
        'tssb_low_resource_10_shot_42_seed': read_tfix_examples,
        'tssb_low_resource_10_shot_87_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_100_shot_100_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_100_shot_13_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_100_shot_21_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_100_shot_42_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_100_shot_87_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_50_shot_100_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_50_shot_13_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_50_shot_21_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_50_shot_42_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_50_shot_87_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_10_shot_100_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_10_shot_13_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_10_shot_21_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_10_shot_42_seed': read_tfix_examples,
        'joint_upsampling_tssb_low_resource_10_shot_87_seed': read_tfix_examples,
        'tssb_joint_100_shot': read_tfix_examples,
        'tssb_joint_50_shot': read_tfix_examples,
        'tssb_joint_10_shot': read_tfix_examples

    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
