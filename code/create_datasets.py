import os
import json
import argparse
import itertools
import collections
import numpy as np
import time

# simple formal language dataset: (subset of) all combinations with optional restrictions on tokens for positions
class Dataset():
    
    def __init__(self,
                 vocabulary=['a', 'b'], # tokens to concatenate to sequences
                 min_len=1, # smallest sequence
                 max_len=0, # longest sequence; len(vocabulary) if 0
                 voc_in_position={}, # dict from positions to tokens only allowed in those positions
                 discard_from_position={}, # dict from positions to list of tokens in vocabulary: don't put these tokens in these positions
                 dont_repeat=[], # subset of vocabulary to not repeat in same str
                 repeat=[], # subset of vocabulary to always repeat in same str
                 max_size=10000): # maximum number of datapoints in resulting dataset)

        sequences = []
        max_len = len(vocabulary) if not max_len else max_len
        
        # make vocabulary and restrictions available for set operations
        vocabulary = set(vocabulary)
        voc_in_position = {p:set(voc_in_position[p]) for p in voc_in_position}
        discard_from_position = collections.defaultdict(set, {p:set(discard_from_position[p]) for p in discard_from_position})
        
        # how many combinations would be available
        max_comb = 0
        for n in range(min_len, max_len+1):
            max_comb += len(vocabulary)**n
            if max_comb > max_size:
                break
        
        # if all possible combinations don't exceed max_size, do them all
        if max_comb <= max_size:
            
            for n in range(min_len, max_len+1): # do sequences of each size between min_len and max_len
                n_seq = []
                
                for i in range(n): # all tokens for each position in the sequence
                    pos_voc = voc_in_position[i] if i in voc_in_position else vocabulary
                    pos_voc -= discard_from_position[i] # don't use tokens discarded in the current position
                    n_seq.append(list(pos_voc))
                
                n_seq = list(itertools.product(*n_seq))
                sequences += [''.join(seq) for seq in n_seq]
                
        # if all possible combinations exceed max_size, do random selection of size max_size
        else:        
            sequences = set() # use set for avoiding repetition of randomly generated datapoints
            lengths = list(range(min_len, max_len+1)) # possible sequence lengths
            
            while len(sequences) < max_size: # add new sequence until max_size is reached
                n = np.random.choice(lengths) # choose random sequence length
                n_seq_rand = []
                
                for i in range(n): # random token for each position in the sequence
                    pos_voc = voc_in_position[i] if i in voc_in_position else vocabulary
                    pos_voc -= discard_from_position[i]
                    rand_token = np.random.choice((list(pos_voc)))
                    n_seq_rand.append(rand_token)
                    
                sequences.add(''.join(n_seq_rand))
                
            sequences = list(sequences)
            
        # no repetition of tokens in "dont_repeat"
        if dont_repeat:
            for i,seq in enumerate(sequences):
                seq = list(seq)
                for c in dont_repeat:
                    while seq.count(c) > 1:
                        seq.remove(c)
                while len(seq) < min_len:
                    can_add = set(vocabulary) - (set(seq) & set(dont_repeat))
                    if not can_add:
                        print('Contradictory requirements for "min_len" and "dont_repeat"')
                        return
                    add_position = np.random.choice(list(range(len(seq))))
                    seq = seq[:add_position] + [np.random.choice(list(can_add))] + seq[add_position:]
                sequences[i] = ''.join(seq)
            sequences = list(set(sequences))
        
        # repetition of some token in "repeat" in each src datapoint
        if repeat:
            for i,seq in enumerate(sequences):
                seq = list(seq)
                if max([seq.count(c) for c in repeat]) < 2:
                    to_repeat = np.random.choice(list(repeat))
                    while seq.count(to_repeat) < 2:
                        position = np.random.choice(range(len(seq)+1))
                        seq = seq[:position] + [to_repeat] + seq[position:]
                    while len(seq) > max_len:
                        can_delete = [c for c in seq if c!=to_repeat or seq.count(c)>2]
                        if not can_delete:
                            print('Contradictory requirements for "max_len" and "repeat"')
                            return
                        seq.remove(np.random.choice(can_delete))
                    sequences[i] = ''.join(seq)
            sequences = list(set(sequences))
            
        self.src = sorted(sequences)
        self.vocabulary = vocabulary
        self.min_len = min_len
        self.max_len = max_len
        self.voc_in_position = voc_in_position
        self.discard_from_position = discard_from_position
        self.max_size = max_size
    
    # set target values
    def set_tgt(self,
                task='copy', # copy/different/reverse/uppercase/count/repeat/custom
                custom_label='X', # label in custom task
                pad_to=0,
                pad_str='P',
                randomize_pad=False): 
        
        assert task in ['copy', 'different', 'reverse', 'replace', 'uppercase', 'count', 'repeat', 'same_size', 'different_size', 'custom']
        
        # sequences mapped to themselves
        if task=='copy':
            self.tgt = self.src
        
        # sequences mapped to different sequences with edit distance between 1 and max_edit_distance
        elif task == 'different':
            if len(self.src) < 2:
                print('At least 2 source expressions needed to make targets')
                return
            self.tgt = []
            for seq in self.src:
                tgt_seq = np.random.choice(self.src)
                while seq == tgt_seq:
                    tgt_seq = np.random.choice(self.src)
                self.tgt.append(tgt_seq)
            assert [s!=t for (s,t) in zip(self.src, self.tgt)] # just a sanity check
        
        # sequences mapped to their reversal
        elif task=='reverse':
            self.tgt = [seq[::-1] for seq in self.src]
                
        # sequences mapped to uppercase variants
        elif task=='uppercase':
            self.tgt = [seq.upper() for seq in self.src]
        
        # sequqneces mapped to token counts
        elif task=='count':
            self.tgt = [str(len(seq)) for seq in self.src]
        
        elif task=='repeat':
            self.tgt = [str(int(len(seq)!=len(set(seq)))) for seq in self.src]
        
        elif task=='same_size':
            self.tgt = [np.random.choice([s for s in self.src if len(s)==len(seq)]) for seq in self.src]
        
        elif task=='different_size':
            if len(set([len(seq) for seq in self.src])) == 1:
                print('At least two lenghts needed in source to make targets')
                return
            self.tgt = [np.random.choice([s for s in self.src if len(s)!=len(seq)]) for seq in self.src]
        
        # sequences mapped to user-given custom label
        elif task=='custom':
            self.tgt = [str(custom_label) for seq in self.src]
        
        assert len(self.src) == len(self.tgt) # just a sanity check
        
        # padding
        if pad_to:
            for i,s in enumerate(self.src):
                s_pad_add = max(0, pad_to - len(s))
                if randomize_pad and s_pad_add:
                    s_pad_ix = sorted(np.random.choice(list(range(pad_to)), s_pad_add, replace=False))
                    s_padded = ''
                    for j in range(pad_to):
                        if j in s_pad_ix:
                            s_padded += pad_str
                        else:
                            s_padded += s[0]
                            s = s[1:]
                    self.src[i] = s_padded
                elif s_pad_add:
                    s_padded = s + ''.join([pad_str for j in range(s_pad_add)])
                    self.src[i] = s_padded

# specify dataset params and labels, create dataset, save to file
def main(args):
    
    time0 = time.time()
    
    # set random seed if specified
    if args.random_seed:
        np.random.seed(args.random_seed)
    
    # make dataset
    ds = Dataset(vocabulary=args.vocabulary,
                 min_len=args.min_len,
                 max_len=args.max_len,
                 dont_repeat=args.dont_repeat,
                 repeat=args.repeat,
                 voc_in_position=args.voc_in_position,
                 discard_from_position=args.discard_from_position,
                 max_size=args.max_size)
    
    # make targets
    ds.set_tgt(task=args.task,
               custom_label=args.custom_label,
               pad_to=args.pad_to,
               pad_str=args.pad_str,
               randomize_pad=args.randomize_pad)
    
    # make src-tgt pairs
    src_tgt = list(zip(ds.src, ds.tgt))
    
    np.random.shuffle(src_tgt)

    # save dataset parameters to json file
    params = {'task':args.task,
              'vocabulary':args.vocabulary,
              'min_len':args.min_len,
              'max_len':args.max_len,
              'voc_in_position':args.voc_in_position,
              'discard_from_position':args.discard_from_position,
              'random_seed':args.random_seed,
              'size':len(src_tgt)}    

    # make train-eval split
    train, val = [], []
    if args.eval_split:
        eval_split = round(args.eval_split * len(src_tgt))
        train, val = src_tgt[eval_split:], src_tgt[:eval_split]
        params['train_size'] = len(train)
        params['eval_size'] = len(val)
    
    # save folder name = task + vocabulary[:max_fname_len] + optional differentiating int
    save_folder = os.path.join(args.save_folder, args.task, ''.join(args.vocabulary[:args.max_fname_len]))
    cont_str = '...' if len(ds.vocabulary) > args.max_fname_len else ''
    save_folder = save_folder + cont_str
    
    # If task is 'count', add numbers to count to save folder name
    if args.task == 'count':
        save_folder = '{0}_{1}-{2}'.format(save_folder, ds.min_len, ds.max_len)  

    if args.save_suffix:
        save_folder += '_' + args.save_suffix
    
    # add int to save_folder name for differentiating between variants
    if args.dont_overwrite:
        folder_int = 1
        while os.path.exists('{0}_{1}'.format(args.save_folder, folder_int)):
            folder_int += 1
        save_folder = '{0}_{1}'.format(save_folder, folder_int)
    
    print('Creating dataset for', save_folder, end=':')
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    with open(os.path.join(save_folder, 'params.json'), 'w') as f:
        json.dump(params, f)
    
    # save data to a txt-file: tokens separated by separator, src and tgt separated by delimiter
    with open(os.path.join(save_folder, 'all.txt'), 'w') as f:
        for src, tgt in src_tgt:
            f.write(args.separator.join(src) + args.delimiter + args.separator.join(tgt) + '\n') # add separators and delimiters
    
    if train!=[] and val!=[]:
        with open(os.path.join(save_folder, 'train.txt'), 'w') as f:
            for src, tgt in train:
                f.write(args.separator.join(src) + args.delimiter + args.separator.join(tgt) + '\n') # add separators and delimiters
        with open(os.path.join(save_folder, 'eval.txt'), 'w') as f:
            for src, tgt in val:
                f.write(args.separator.join(src) + args.delimiter + args.separator.join(tgt) + '\n') # add separators and delimiters
    
    print(' took', round(time.time()-time0, 2), 'seconds')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-task', '--task', required=True) # copy/different/reverse/count/repeat/custom
    arg_parser.add_argument('-voc', '--vocabulary', type=list, required=True)
    arg_parser.add_argument('--dont_repeat', type=list, default=[]) # # subset of vocabulary not to repeat in same str
    arg_parser.add_argument('--repeat', type=list, default=[]) # subset of vocabulary to always repeat in same str
    arg_parser.add_argument('--min_len', type=int, default=1)
    arg_parser.add_argument('--max_len', type=int, default=0) # vocabulary length by default
    arg_parser.add_argument('--voc_in_position', type=dict, default={})
    arg_parser.add_argument('--discard_from_position', type=dict, default={})
    arg_parser.add_argument('--custom_label', default='X')
    arg_parser.add_argument('--max_size', type=int, default=10000)
    arg_parser.add_argument('--random_seed', type=int, default=12345) # set random seed
    arg_parser.add_argument('--separator', default=' ') # separates tokens in saved txt-file
    arg_parser.add_argument('--delimiter', default='\t') # separates src from tgt in saved txt-file
    arg_parser.add_argument('--pad_to', type=int, default=0) # pad to same size
    arg_parser.add_argument('--pad_str', default='P') # char to use for padding
    arg_parser.add_argument('--randomize_pad', action='store_true') # pad in random positions instead of end
    arg_parser.add_argument('--eval_split', type=float, default=0.2) # train-eval split
    arg_parser.add_argument('--save_folder', default='data/')
    arg_parser.add_argument('--save_suffix', default='') # add to save_folder name
    arg_parser.add_argument('--dont_overwrite', action='store_true') # add differentiating int to folder
    arg_parser.add_argument('--max_fname_len', type=int, default=100)
    args = arg_parser.parse_args()
    main(args)