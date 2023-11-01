# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import torch
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist
import higher

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_batch_size = 2 * args.train_batch_size
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            elif args.model_type == 'unixcoder':
                loss, _, _ = model(source_ids=source_ids, target_ids=target_ids)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl

def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):

    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)

    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        if args.model_type == 'unixcoder':
            source_ids = batch[0].to(args.device)
        else:
            source_ids = batch[0].to(args.device)
            source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            elif args.model_type == 'unixcoder':
                preds = model(source_ids)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))


    dev_accs, predictions = [], []
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        for pred_nl, gold in zip(pred_nls, eval_examples):
            if 'tfix' in args.task or 'sb4j' in args.task:
                pred_nl = ' '.join(pred_nl.split())
                gold.target = ' '.join(gold.target.split())
                gold.source = ' '.join(gold.source.split())

            dev_accs.append(pred_nl.strip() == gold.target.strip())

            f.write(pred_nl.strip() + '\n')
            f1.write(gold.target.strip() + '\n')
            f2.write(gold.source.strip() + '\n')


    bleu = round(_bleu(gold_fn, output_fn), 2)

    em = np.mean(dev_accs) * 100
    result = {'em': em, 'bleu': bleu}
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def main():
    tasks_high_resource_tfix = ['no-invalid-this',
                                'no-undef',
                                'no-unused-vars',
                                'comma-style',
                                'no-redeclare',
                                'no-extra-semi',
                                'no-unreachable',
                                'prefer-rest-params',
                                'no-debugger',
                                'no-throw-literal',
                                'guard-for-in',
                                'no-console',
                                'no-useless-escape',
                                'prefer-spread',
                                'no-dupe-keys',
                                'no-empty',
                                'no-process-exit',
                                'no-cond-assign',
                                'no-extra-boolean-cast',
                                'generator-star-spacing',
                                'no-constant-condition']
    tasks_high_resource_sb4j = ['CHANGE_IDENTIFIER',
                                'OVERLOAD_METHOD_MORE_ARGS',
                                'CHANGE_NUMERAL',
                                'CHANGE_MODIFIER',
                                'MORE_SPECIFIC_IF',
                                'CHANGE_OPERATOR']
    tasks_high_resource_tssb = ['SINGLE_STMT',
                                'CHANGE_STRING_LITERAL',
                                'CHANGE_IDENTIFIER_USED',
                                'CHANGE_BINARY_OPERAND',
                                'SAME_FUNCTION_MORE_ARGS',
                                'WRONG_FUNCTION_NAME',
                                'CHANGE_NUMERIC_LITERAL',
                                'ADD_FUNCTION_AROUND_EXPRESSION',
                                'CHANGE_ATTRIBUTE_USED',
                                'SINGLE_TOKEN',
                                'ADD_METHOD_CALL',
                                'MORE_SPECIFIC_IF',
                                'ADD_ELEMENTS_TO_ITERABLE',
                                'SAME_FUNCTION_LESS_ARGS']
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.zero_grad()
    model.to(args.device)
    model.train()

    pool = multiprocessing.Pool(args.cpu_cont)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')
    fb = open(os.path.join(args.output_dir, 'eval_summary.log'), 'a+')
    fc = open(os.path.join(args.output_dir, 'test_summary.log'), 'a+')

    if args.do_train:
        logger.info("***** Running meta train *****")
        if args.local_rank in [-1, 0]:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)
        if args.task == 'tfix_high_resource_meta_crossfit':
            temp_task = 'tfix_high_resource'
            tasks_majority = tasks_high_resource_tfix
        elif args.task == 'sb4j_high_resource_meta_crossfit':
            temp_task = 'sb4j_high_resource'
            tasks_majority = tasks_high_resource_sb4j
        elif args.task == 'tssb_high_resource_meta_crossfit':
            temp_task = 'tssb_high_resource'
            tasks_majority = tasks_high_resource_tssb
        temp_train_dir = '{}/{}'.format(args.data_dir, temp_task)
        temp_train_filename = '{}/train.pkl'.format(temp_train_dir)
        logger.info("  Overall train file is: {}".format(temp_train_filename))
        train_examples, train_data = load_and_cache_gen_data(args, temp_train_filename, pool, tokenizer,
                                                             'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if args.n_gpu > 1:
            # for DataParallel
            model = torch.nn.DataParallel(model)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        meta_chk = []
        if args.do_maml:
            meta_chk.append('MAML')
            logger.info("Meta training with {}".format(meta_chk[0]))
        elif args.do_fomaml:
            meta_chk.append('FOMAML')
            logger.info("Meta training with {}".format(meta_chk[0]))
        elif args.do_reptile:
            meta_chk.append('Reptile')
            logger.info("Meta training with {}".format(meta_chk[0]))
        else:
            assert len(
                meta_chk) != 0, "Please check the selected meta learning algorithm, it should either one in [MAML, FOMAML, Reptile]"
        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_em, best_test_em, best_ppl = 0, -1, -1, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            model.train()
            if args.do_reptile:
                mlg = 'Reptile'
                if args.task == 'tfix_high_resource_meta_crossfit':
                    args.task = 'tfix_high_resource'
                elif args.task == 'sb4j_high_resource_meta_crossfit':
                    args.task = 'sb4j_high_resource'
                elif args.task == 'tssb_high_resource_meta_crossfit':
                    args.task = 'tssb_high_resource'

                train_dir = '{}/{}'.format(args.data_dir, args.task)
                args.train_filename = '{}/train.pkl'.format(train_dir)
                logger.info("  Train file is: {}".format(args.train_filename))
                train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer,
                                                                     'train')
                train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                              num_workers=4, pin_memory=True)
                # Start training
                train_example_num = len(train_data)
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", train_example_num)
                logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
                bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
                nb_tr_examples, nb_tr_steps, tr_loss, logging_loss = 0, 0, 0, 0
                for step, batch in enumerate(bar):
                    a, b = batch
                    if int(a.size(0)) > int((args.train_batch_size) / 2):
                        batch = tuple(t.to(args.device) for t in batch)
                        n_meta_lr = int((args.train_batch_size) / 2)
                        source_ids, target_ids = batch
                        source_mask = source_ids.ne(tokenizer.pad_token_id)
                        target_mask = target_ids.ne(tokenizer.pad_token_id)
                        inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_learning_rate)
                        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (
                                fast_model, diffopt):
                            if args.model_type == 'roberta':
                                loss, _, _ = fast_model(source_ids=source_ids[n_meta_lr:],
                                                        source_mask=source_mask[n_meta_lr:],
                                                        target_ids=target_ids[n_meta_lr:],
                                                        target_mask=target_mask[n_meta_lr:])
                            elif args.model_type == 'unixcoder':
                                loss, _, _ = fast_model(source_ids=source_ids[n_meta_lr:],
                                                        target_ids=target_ids[n_meta_lr:])
                            else:
                                outputs = fast_model(input_ids=source_ids[n_meta_lr:],
                                                     attention_mask=source_mask[n_meta_lr:],
                                                     labels=target_ids[n_meta_lr:],
                                                     decoder_attention_mask=target_mask[n_meta_lr:])
                                loss = outputs.loss
                            if args.model_type in ['unixcoder']:
                                diffopt.step(loss.detach().requires_grad_())
                            elif args.model_type in ['roberta']:
                                diffopt.step(loss.detach().requires_grad_())
                            else:
                                diffopt.step(loss, grad_callback=lambda grads: [g.detach() for g in grads])
                            if args.model_type == 'roberta':
                                qry_loss, _, _ = fast_model(source_ids=source_ids[:n_meta_lr],
                                                            source_mask=source_mask[:n_meta_lr],
                                                            target_ids=target_ids[:n_meta_lr],
                                                            target_mask=target_mask[:n_meta_lr])
                            elif args.model_type == 'unixcoder':
                                qry_loss, _, _ = fast_model(source_ids=source_ids[:n_meta_lr],
                                                            target_ids=target_ids[:n_meta_lr])
                            else:
                                outputs = fast_model(input_ids=source_ids[:n_meta_lr],
                                                     attention_mask=source_mask[:n_meta_lr],
                                                     labels=target_ids[:n_meta_lr],
                                                     decoder_attention_mask=target_mask[:n_meta_lr])

                                qry_loss = outputs.loss
                            if args.n_gpu > 1:
                                qry_loss = qry_loss.mean()  # mean() to average on multi-gpu.
                            if args.gradient_accumulation_steps > 1:
                                qry_loss = qry_loss / args.gradient_accumulation_steps

                            nb_tr_examples += source_ids.size(0)
                            nb_tr_steps += 1

                            try:
                                qry_loss.backward()
                            except:
                                logger.info("Error: Backward()")

                            tr_loss += qry_loss.item()

                            if nb_tr_steps % args.gradient_accumulation_steps == 0:
                                global_step += 1

                    if nb_tr_steps % args.task_batch_size == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()

                    if nb_tr_steps % args.logging_steps == 0:
                        tb_writer.add_scalar('lr', args.learning_rate, global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps,
                                             global_step)
                        logging_loss = tr_loss
                        bar.set_description(
                            "[{}] Meta-learning train loss {} with meta learning algo {}".format(cur_epoch, round(
                                tr_loss / global_step, 3), mlg))
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
            else:
                for sub_task in tasks_majority:
                    args.sub_task = sub_task
                    data_dir = '{}/{}/{}'.format(args.data_dir, args.task, args.sub_task)
                    args.train_filename = '{}/train.pkl'.format(data_dir)
                    logger.info("  Train file is: {}".format(args.train_filename))
                    # Prepare training data loader
                    train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer,
                                                                         'train')
                    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(
                        train_data)
                    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                                  num_workers=4, pin_memory=True)
                    # Start training
                    train_example_num = len(train_data)
                    logger.info("***** Running training on {}*****".format(sub_task))
                    logger.info("  Num examples = %d", train_example_num)
                    logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))

                    bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
                    nb_tr_examples, nb_tr_steps, tr_loss, logging_loss = 0, 0, 0, 0
                    for step, batch in enumerate(bar):
                        a,b = batch
                        if int(a.size(0)) > int((args.train_batch_size) / 2):
                            batch = tuple(t.to(args.device) for t in batch)
                            n_meta_lr = int((args.train_batch_size) / 2)

                            source_ids, target_ids = batch
                            source_mask = source_ids.ne(tokenizer.pad_token_id)
                            target_mask = target_ids.ne(tokenizer.pad_token_id)
                            inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_learning_rate)
                            if args.do_maml:
                                mlg = 'MAML'
                                with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fast_model, diffopt):
                                    if args.model_type == 'roberta':
                                        loss, _, _ = fast_model(source_ids=source_ids[n_meta_lr:], source_mask=source_mask[n_meta_lr:],
                                                           target_ids=target_ids[n_meta_lr:], target_mask=target_mask[n_meta_lr:])
                                    elif args.model_type == 'unixcoder':
                                        loss, _, _ = fast_model(source_ids=source_ids[n_meta_lr:],
                                                                target_ids=target_ids[n_meta_lr:])
                                    else:
                                        outputs = fast_model(input_ids=source_ids[n_meta_lr:], attention_mask=source_mask[n_meta_lr:],
                                                        labels=target_ids[n_meta_lr:], decoder_attention_mask=target_mask[n_meta_lr:])
                                        loss = outputs.loss

                                    diffopt.step(loss)

                                    if args.model_type == 'roberta':
                                        qry_loss, _, _ = fast_model(source_ids=source_ids[:n_meta_lr], source_mask=source_mask[:n_meta_lr],
                                                           target_ids=target_ids[:n_meta_lr], target_mask=target_mask[:n_meta_lr])
                                    elif args.model_type == 'unixcoder':
                                        qry_loss, _, _ = fast_model(source_ids=source_ids[:n_meta_lr],
                                                                    target_ids=target_ids[:n_meta_lr])
                                    else:
                                        outputs = fast_model(input_ids=source_ids[:n_meta_lr], attention_mask=source_mask[:n_meta_lr],
                                                        labels=target_ids[:n_meta_lr], decoder_attention_mask=target_mask[:n_meta_lr])

                                        qry_loss = outputs.loss
                                    if args.n_gpu > 1:
                                        qry_loss = qry_loss.mean()  # mean() to average on multi-gpu.
                                    if args.gradient_accumulation_steps > 1:
                                        qry_loss = qry_loss / args.gradient_accumulation_steps

                                    nb_tr_examples += source_ids.size(0)
                                    nb_tr_steps += 1

                                    try:
                                        qry_loss.backward()
                                    except:
                                        logger.info("Error: Backward()")

                                    tr_loss += qry_loss.item()

                                    if nb_tr_steps % args.gradient_accumulation_steps == 0:
                                        global_step += 1

                        if nb_tr_steps % args.task_batch_size == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                            model.zero_grad()

                        if nb_tr_steps % args.task_batch_size == 0:
                            tb_writer.add_scalar('lr', args.learning_rate, global_step)
                            tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.task_batch_size,
                                                 global_step)
                            logging_loss = tr_loss
                            bar.set_description(
                                "[{}] Meta-learning train loss {} on error type {} with meta learning algo {}".format(
                                    cur_epoch, round(tr_loss / global_step, 3), sub_task, mlg))
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()
            if args.do_eval:
                logger.info("  " + "***** In epoch eval PPL *****")
                logger.info("  Batch size = %d", args.eval_batch_size)
                if 'tfix_high_resource' in args.task:
                    eval_dir = '{}/{}'.format(args.data_dir, 'tfix_high_resource')
                elif 'sb4j_high_resource' in args.task:
                    eval_dir = '{}/{}'.format(args.data_dir, 'sb4j_high_resource')
                elif 'tssb_high_resource' in args.task:
                    eval_dir = '{}/{}'.format(args.data_dir, 'tssb_high_resource')
                args.dev_filename = '{}/val.pkl'.format(eval_dir)
                logger.info("  Eval file is: {}".format(args.dev_filename))
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       is_sample=True)
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.do_eval_bleu:
                    logger.info("  " + "***** In epoch eval EM & BLEU *****")
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    if 'tfix_high_resource' in args.task:
                        eval_dir = '{}/{}'.format(args.data_dir, 'tfix_high_resource')
                    elif 'sb4j_high_resource' in args.task:
                        eval_dir = '{}/{}'.format(args.data_dir, 'sb4j_high_resource')
                    elif 'tssb_high_resource' in args.task:
                        eval_dir = '{}/{}'.format(args.data_dir, 'tssb_high_resource')
                    args.dev_filename = '{}/val.pkl'.format(eval_dir)
                    logger.info("  Eval file is: {}".format(args.dev_filename))
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=True)

                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    fb.write("[%d] Eval bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                        cur_epoch, dev_bleu+dev_em, dev_bleu, dev_em))
                    dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        tb_writer.add_scalar('dev_bleu', dev_bleu, cur_epoch)
                        tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if cur_epoch%5 == 0:
                        logger.info("  [%d] Epoch bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        output_dir = os.path.join(args.output_dir, 'checkpoint-epoch-' + str(cur_epoch))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the epoch %d model into %s", cur_epoch, output_model_file)
                        if 'sb4j' in args.task and args.do_test_bleu:
                            logger.info("  " + "***** In epoch testing EM & BLEU *****")
                            logger.info("  Batch size = %d", args.eval_batch_size)
                            if 'sb4j_high_resource' in args.task:
                                test_dir = '{}/{}'.format(args.data_dir, 'sb4j_high_resource')
                            args.test_filename = '{}/test.pkl'.format(test_dir)
                            logger.info("  Test file is: {}".format(args.test_filename))
                            test_examples, test_data = load_and_cache_gen_data(args, args.test_filename, pool,
                                                                               tokenizer,
                                                                               'test',
                                                                               only_src=True, is_sample=False)
                            result = eval_bleu_epoch(args, test_data, test_examples, model, tokenizer, 'test',
                                                     'e%d' % cur_epoch)
                            test_bleu, test_em = result['bleu'], result['em']
                            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
                            result_str = "[%d test] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (
                                cur_epoch, test_bleu, test_em, test_codebleu)
                            logger.info(result_str)
                            fc.write(result_str)

                            if best_test_em < test_em:
                                best_test_em = test_em
                                fa.write("[%d test] Best test em changed into %.2f\n" % (cur_epoch, test_em))
                                output_dir = os.path.join(args.output_dir, 'checkpoint-best-test-em')
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                if args.data_num == -1 or args.always_save_model:
                                    model_to_save = model.module if hasattr(model, 'module') else model
                                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                                    torch.save(model_to_save.state_dict(), output_model_file)
                                    logger.info("Save the best test em model into %s", output_model_file)
                        if 'tssb' in args.task and args.do_test_bleu:
                            logger.info("  " + "***** In epoch testing EM & BLEU *****")
                            logger.info("  Batch size = %d", args.eval_batch_size)
                            if 'tssb_high_resource' in args.task:
                                test_dir = '{}/{}'.format(args.data_dir, 'tssb_high_resource')
                            args.test_filename = '{}/test.pkl'.format(test_dir)
                            logger.info("  Test file is: {}".format(args.test_filename))
                            test_examples, test_data = load_and_cache_gen_data(args, args.test_filename, pool,
                                                                               tokenizer,
                                                                               'test',
                                                                               only_src=True, is_sample=False)
                            result = eval_bleu_epoch(args, test_data, test_examples, model, tokenizer, 'test',
                                                     'e%d' % cur_epoch)
                            test_bleu, test_em = result['bleu'], result['em']
                            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
                            result_str = "[%d test] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (
                                cur_epoch, test_bleu, test_em, test_codebleu)
                            logger.info(result_str)
                            fc.write(result_str)

                            if best_test_em < test_em:
                                best_test_em = test_em
                                fa.write("[%d test] Best test em changed into %.2f\n" % (cur_epoch, test_em))
                                output_dir = os.path.join(args.output_dir, 'checkpoint-best-test-em')
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                if args.data_num == -1 or args.always_save_model:
                                    model_to_save = model.module if hasattr(model, 'module') else model
                                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                                    torch.save(model_to_save.state_dict(), output_model_file)
                                    logger.info("Save the best test em model into %s", output_model_file)
                        if 'tfix' in args.task and args.do_test_bleu:
                            logger.info("  " + "***** In epoch testing EM & BLEU *****")
                            logger.info("  Batch size = %d", args.eval_batch_size)
                            if 'tfix_high_resource' in args.task:
                                test_dir = '{}/{}'.format(args.data_dir, 'tfix_high_resource')
                            args.test_filename = '{}/test.pkl'.format(test_dir)
                            logger.info("  Test file is: {}".format(args.test_filename))
                            test_examples, test_data = load_and_cache_gen_data(args, args.test_filename, pool,
                                                                               tokenizer,
                                                                               'test',
                                                                               only_src=True, is_sample=False)
                            result = eval_bleu_epoch(args, test_data, test_examples, model, tokenizer, 'test',
                                                     'e%d' % cur_epoch)
                            test_bleu, test_em = result['bleu'], result['em']
                            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
                            result_str = "[%d test] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (
                                cur_epoch, test_bleu, test_em, test_codebleu)
                            logger.info(result_str)
                            fc.write(result_str)

                            if test_em > best_test_em:
                                best_test_em = test_em
                                fa.write("[%d test] Best test em changed into %.2f\n" % (cur_epoch, test_em))
                                output_dir = os.path.join(args.output_dir, 'checkpoint-best-test-em')
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                if args.data_num == -1 or args.always_save_model:
                                    model_to_save = model.module if hasattr(model, 'module') else model
                                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                                    torch.save(model_to_save.state_dict(), output_model_file)
                                    logger.info("Save the best test em model into %s", output_model_file)

                    if dev_em > best_em:
                        logger.info("  [%d] Best em: %.2f ", cur_epoch, dev_em)
                        logger.info("  " + "*" * 20)
                        best_em = dev_em
                        fa.write("[%d] Best em changed into %.2f \n" % (cur_epoch, dev_em))
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-dev-em')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best em model into %s", output_model_file)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break


            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        if 'tfix_high_resource' in args.task:
            test_dir = '{}/{}'.format(args.data_dir, 'tfix_high_resource')
        elif 'sb4j_high_resource' in args.task:
            test_dir = '{}/{}'.format(args.data_dir, 'sb4j_high_resource')
        elif 'tssb_high_resource' in args.task:
            test_dir = '{}/{}'.format(args.data_dir, 'tssb_high_resource')
        args.test_filename = '{}/test.pkl'.format(test_dir)
        logger.info("  Test file is: {}".format(args.test_filename))
        for criteria in ['best-bleu', 'last', 'best-dev-em', 'best-test-em']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            test_bleu, test_em = result['bleu'], result['em']
            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
            result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
            logger.info(result_str)
            fa.write(result_str)
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write(result_str)

        logger.info("Finish and take {}".format(get_elapse_time(t0)))
        fa.write("Finish and take {}\n".format(get_elapse_time(t0)))
        fa.close()


if __name__ == "__main__":
    main()
