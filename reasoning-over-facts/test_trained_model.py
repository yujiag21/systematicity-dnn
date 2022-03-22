import argparse
import glob
import json
import logging
import os

import random

from pathlib import Path
from typing import Dict, List, Tuple
from glob import glob

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)

import config

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def batchify_dict(d, args, tokenizer):
    masked_queries = [f'{query} {tokenizer.mask_token}' for query in d]
    masked_queries = [torch.Tensor(seq) for seq in tokenizer.batch_encode_plus(masked_queries)['input_ids']]
    masked_queries = pad_sequence(masked_queries, batch_first=True, padding_value=tokenizer.pad_token_id)
    batches = np.split(masked_queries, list(range(0, len(masked_queries), args.batch_size))[1:])
    return d, batches

def evaluate(args, corrects, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    def get_mask_idx(batch):
        # determine which one the masked y
        mask_token = tokenizer.mask_token_id
        return [list(batch[i]).index(mask_token) for i in range(batch.shape[0])]

    def compute_ranked_accuracy(query2answers):

        accurate = 0
        tp = 0
        fp = 0
        fn = 0
        total = 0

        answers, batches = query2answers
        for batch in tqdm(batches, desc="Testing"):
            batch = torch.tensor(batch).to(torch.int64)
            batch = batch.to(args.device)
            prediction_scores = model(batch)[0]
            # print('prediction_scores', [tokenizer.convert_ids_to_tokens(prediction_score) for prediction_score in prediction_scores])
            masked_indices = get_mask_idx(batch) # determine which one the masked y(index)
            prediction_scores = prediction_scores[np.arange(prediction_scores.shape[0]), masked_indices, :] # seperate the y from other.
            # print('prediction_scores', [tokenizer.convert_ids_to_tokens(prediction_score) for prediction_score in prediction_scores])

            for i, (prediction, sample) in enumerate(zip(prediction_scores, batch)):
                key = " ".join(tokenizer.convert_ids_to_tokens(sample[1:masked_indices[i]]))
                print('key',key)
                correct_objects = answers[key]
                numb_correct_answers = len(correct_objects)
                predicted_ids = torch.argsort(prediction, dim=0, descending=True)[:numb_correct_answers]
                ranked_predictions = tokenizer.convert_ids_to_tokens(predicted_ids)
                with open(os.path.join(args.output_dir, "test_result.txt"), "a+") as f:
                    f.write("x:{}, ".format(key))
                    f.write("correct_y:{}, ".format(correct_objects))
                    f.write("predictions:{}\n ".format(ranked_predictions))

                    print('x:', key)
                    print('predictions:', ranked_predictions)
                    print('correct_objects:', correct_objects)

                accurate += len(set(ranked_predictions) & set(correct_objects)) / numb_correct_answers
                total += 1.0
                if ranked_predictions[0]=="true" and correct_objects[0]=="true":
                    tp += 1
                if ranked_predictions[0]=="true" and correct_objects[0]=="false":
                    fp += 1
                if ranked_predictions[0]=="false" and correct_objects[0]=="true":
                    fn += 1
                print('tp',tp)
                print('fp', fp)
                print('fn', fn)

                precision = 0 if tp+fp==0 else tp / (tp + fp)
                recall= 0 if tp+fn==0 else tp / (tp + fn)
                F_score = (2 * recall * precision / (recall + precision)) if (recall + precision)>0 else 0

        return accurate / total, precision, recall, F_score

    model.eval()
    result = {}
    logger.info("***** Running test {} *****".format(prefix))
    logger.info("  Batch size = %d", args.batch_size)

    for eval_type, query2answers in corrects.items():
        with torch.no_grad():
            with open(os.path.join(args.output_dir, "test_result.txt"), "a+") as f:
                f.write("***** Test results {} *****".format(eval_type))
            accuracy, precision, recall, F_score = compute_ranked_accuracy(query2answers)
            # accuracy = compute_ranked_accuracy(query2answers)
            accuracy = round(accuracy, 4)
            precision = round(precision, 4)
            recall = round(recall, 4)
            F_score = round(F_score, 4)
            result[eval_type + '_ranked_acc'] = accuracy
            result[eval_type + '_ranked_precision'] = precision
            result[eval_type + '_ranked_recall'] = recall
            result[eval_type + '_ranked_f_score'] = F_score

    logger.info("***** Test results {} *****".format(prefix))
    with open(os.path.join(args.output_dir, "test_result.txt"), "a+") as f:
        for key in sorted(result.keys()):
            f.write("{} = {}\n".format(key, str(result[key])))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return result


def get_all_checkpoints_from_dir(output_dir):
    dirs = [x for x in glob( output_dir+ "/checkpoint*/")]
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relation', '-r', type=str, required=True,
                        help=f'relation type that is trained on. Available :{", ".join(config.supported_relations)}')
    parser.add_argument('--dataset_name', '-d', required=True, type=str, help='dataset used for train, eval and vocab')
    parser.add_argument('--output_model_name', '-o', type=str, default='',
                        help='Defaults to dataset_name if not stated.')
    parser.add_argument('--batch_size', type=int, default='1024', help='Default is batch size of 256')
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu number")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device)
    if args.output_model_name == '':
        args.output_model_name = args.dataset_name

    data_dir = Path('data') / args.relation / 'datasets' / args.dataset_name
    args.train_data_file = data_dir / 'train.txt'
    args.tokenizer_name = f'data/{args.relation}/vocab/{args.dataset_name}/'
    args.output_dir = f'output/models/{args.relation}/{args.output_model_name}'

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Load pretrained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
    model_config = BertConfig(vocab_size=tokenizer.vocab_size)


    corrects = {"eval": json.load(open(data_dir / 'subject_relation2object_eval.json', 'r', )),
                "train": json.load(open(data_dir / 'subject_relation2object_train.json', 'r', )),
                "rand_eval": json.load(open(data_dir / 'rand_subject_relation2object_eval.json', 'r', )),
                "rand_train": json.load(open(data_dir / 'rand_subject_relation2object_train.json', 'r', ))
                }

    corrects["eval"] = {fact: [corrects["eval"][fact][0]] for fact in corrects["eval"] if "pattern" in fact}
    corrects["train"] = {fact: [corrects["train"][fact][0]] for fact in corrects["train"] if "pattern" in fact}
    corrects["rand_eval"] = {fact: [corrects["rand_eval"][fact][0]] for fact in corrects["rand_eval"] if "pattern" in fact}
    corrects["rand_train"] = {fact: [corrects["rand_train"][fact][0]] for fact in corrects["rand_train"] if "pattern" in fact}
    print(corrects)
    corrects["eval"].update(corrects["rand_eval"])
    corrects["train"].update(corrects["rand_train"])
    print(corrects)
    # corrects = corrects.pop("rand_eval")
    # corrects = corrects.pop("rand_train")

    for eval_type, d in corrects.items():
        corrects[eval_type] = batchify_dict(d, args, tokenizer)

    # print(args.output_dir)
    # model = BertForMaskedLM.from_pretrained(args.output_dir)
    # tokenizer = BertTokenizer.from_pretrained(args.output_dir)
    # model.to(args.device)

    # Evaluation
    results = {}

    checkpoints = get_all_checkpoints_from_dir(args.output_dir)
    logger.info("Test the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

        model = BertForMaskedLM.from_pretrained(checkpoint)
        model.to(args.device)
        model.eval()
        result = evaluate(args, corrects, model, tokenizer, prefix=prefix)
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)

    with open(os.path.join(args.output_dir, "test_result.txt"), "a+") as f:
        f.writelines("Test the following checkpoints:" + str(checkpoints) + "\n")
        for i in results:
            f.writelines(str(i))
            f.writelines("\n\n")
        f.writelines("\n\n")

    return results

if __name__ == "__main__":
    main()
