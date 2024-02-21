import os
import random
import json
import argparse
import numpy as np
from time import time
from os.path import join, exists
from nltk import sent_tokenize
from datetime import timedelta
import re
import operator

def turn_split(data):
    new_data = []
    lens = []
    for i in range(len(data)):
        lens.append(len(data[i].split()))
    idx = lens.index(max(lens))
    for i in range(len(data)):
        if i == idx:
            s_speaker, s_utt = data[i].split(':', maxsplit=1)
            split_turn = sent_tokenize(s_utt.strip())
            len_split_turn = len(split_turn)
            if len_split_turn <= 1:
                if len(data) > 1:
                    # turn merge instead
                    return turn_merge(data)
                else:
                    new_data.append(data[i])
            else:
                nb_chunks = min(max(2, np.random.poisson(3)), len_split_turn) # split in at least 2 chunks
                idx_splits = np.random.choice(range(1, len_split_turn), size=nb_chunks - 1, replace=False)
                cur_s = s_speaker + ':'
                for j in range(len(split_turn)):
                    sent_j = split_turn[j]
                    if j in idx_splits:
                        new_data.append(cur_s)
                        cur_s = s_speaker + ': ' + sent_j
                    else:
                        cur_s += ' ' + sent_j
                new_data.append(cur_s)
        else:
            new_data.append(data[i])
    return new_data

def turn_merge(data):
    new_data = []
    n_turns = len(data)
    st_idx = random.randint(0, n_turns - 2) # merge at least 2 turns
    merge_turns = max(2, np.random.poisson(3))
    ed_idx = min(n_turns - 1, st_idx + merge_turns - 1)
    new_turn = []
    for i in range(len(data)):
        if i >= st_idx and i <= ed_idx:
            if i == st_idx:
                new_turn.append(data[i])
            else:
                new_turn.append(data[i].split(':', maxsplit=1)[1])
            if i == ed_idx:
                new_data.append(' '.join(new_turn))
        else:
            new_data.append(data[i])
    return new_data

def speaker_mask(data, speaker_mask_ratio):
    for i in range(len(data)):
        if random.random() < speaker_mask_ratio:
            cur = data[i].split(':', maxsplit=1)
            cur[0] = '[MASK] '
            data[i] = ':'.join(cur)
    return data

def text_infilling(data, text_infilling_ratio):
    new_data = []
    for i in range(len(data)):
        s_speaker, s_utt = data[i].split(':', maxsplit=1)
        s_utt = s_utt.strip().split()
        cur = []
        left = 0
        for j in range(len(s_utt)):
            if left > 0:
                left = left - 1
                continue
            if s_utt[j] == '[MASK]':
                cur.append(s_utt[j])
                continue
            if random.random() < text_infilling_ratio:
                left = np.random.poisson(18)
                cur.append('[MASK]')
                if left > 0:
                    left = left - 1
                    continue
            cur.append(s_utt[j])
        new_data.append(s_speaker + ': ' + ' '.join(cur))
    return new_data

def shuffling(data, shuffling_ratio):
    if random.random() < shuffling_ratio:
        random.shuffle(data)
    return data

"""def turn_insertion(data, utterances_insert, speakers_insert, insert_prob):
    new_data = []
    for i in range(len(data)):
        if random.random() < insert_prob:
            span_insert_len = 1 + np.random.binomial(2, 0.5)
            for _ in range(span_insert_len):
                utt_insert = random.choice(utterances_insert)
                spk_insert = random.choice(speakers_insert)
                if ':' not in utt_insert:
                    utt_insert = spk_insert + ': ' + utt_insert
                else:
                    utt_insert_split = utt_insert.split(':', maxsplit=1)
                    utt_insert_split[0] = spk_insert
                    utt_insert = ':'.join(utt_insert_split)
                new_data.append(utt_insert)
        new_data.append(data[i])
    return new_data"""

def turn_insertion(data, utterances_insert, speakers_insert, insert_prob):
    if random.random() < insert_prob:
        span_insert_len = random.randint(2, 5) #TODO
        for _ in range(span_insert_len):
            utt_insert = random.choice(utterances_insert)
            spk_insert = random.choice(speakers_insert)
            if ':' not in utt_insert:
                utt_insert = spk_insert + ': ' + utt_insert
            else:
                utt_insert_split = utt_insert.split(':', maxsplit=1)
                utt_insert_split[0] = spk_insert
                utt_insert = ':'.join(utt_insert_split)
            data.append(utt_insert)
    return data

def drop_utterances(data, nb_drop):
    new_data = []
    # len(data) - 1 because we don't want to drop the last utterance
    idx_drops = random.sample(range(len(data) - 1), k=nb_drop)
    for i in range(len(data)):
        if i not in idx_drops:
            new_data.append(data[i])
    return new_data

def add_noise(data, utterances_insert, nb_spans, avg_span_len, speaker_mask_ratio, text_infilling_ratio, turn_split_prob, insert_prob, shuffling_ratio, drop_utt_prob):
    masked_idx = []
    noised_spans = []
    speakers_dial = [turn.split(':', maxsplit=1)[0] for turn in data if ':' in turn]
    for _ in range(nb_spans):
        span_len = 1 + np.random.binomial(3, 0.5) #TODO

        k = 0
        nb_drop_utt = 0
        while k < span_len:
            drop_current_utt = np.random.binomial(1, drop_utt_prob)
            span_len += drop_current_utt
            nb_drop_utt += drop_current_utt
            k += 1

        if span_len > 0:
            idx_sample = [i for i in range(len(data) - span_len) if all([j not in masked_idx for j in range(i - 1, i + span_len + 1)])]
            #print(masked_idx, len(data), nb_spans, span_len)
            if idx_sample == []:
                #print(f"{k + 1}/{nb_spans}", masked_idx, len(data), span_len, ratio_noised_max, len(masked_idx) / len(data), "\n")
                continue
            idx_start_span = np.random.choice(idx_sample)
            idx_end_span = idx_start_span + span_len
            masked_idx.extend(range(idx_start_span, idx_end_span))
            span_data = data[idx_start_span:idx_end_span].copy()

            # Add MASK speaker to all utterances whose speaker is not defined
            for i in range(len(span_data)):
                utt_i = span_data[i]
                if ':' not in utt_i:
                    span_data[i] = '[MASK] : ' + utt_i

            # dropping utterances at random
            span_data = drop_utterances(span_data, nb_drop_utt)
            # then turn shuffling #TODO
            #span_data = shuffling(span_data, shuffling_ratio)
            # then turn insertion from another dialogue #TODO
            #span_data = turn_insertion(span_data, utterances_insert, speakers_dial, insert_prob)
            # then turn_split or turn_merge
            """if len(span_data) == 1:
                # If there is only one turn in the window, only turn split can be performed
                span_data = turn_split(span_data)
            else:
                # Randomly choose turn split or turn merge
                if random.random() < turn_split_prob:
                    #print("splitsplitsplitsplitsplitsplitsplitsplit\n")
                    span_data = turn_split(span_data)
                else:
                    #print("mergemergemergemergemergemergemergemerge\n")
                    span_data = turn_merge(span_data)"""
            # then speaker mask
            span_data = speaker_mask(span_data, speaker_mask_ratio)
            # then text infilling
            span_data = text_infilling(span_data, text_infilling_ratio)
            # TODO: add utterance deletion, utterance infilling, utterance repetition, change parameters (masking 50% of tokens, shuffling parameter, utterance masking), utterance sentence permutation
            # TODO: tune parameters according to observations
            noised_spans.append((idx_start_span, idx_end_span, span_data))

    #print(masked_idx, len(data), nb_spans)
    noised_spans.sort(key=operator.itemgetter(0))
    all_spans = []
    idx_noised = []
    i = 0
    for n_span in noised_spans:
        if n_span[0] > 0:
            span_str = "\n".join([data[j] for j in range(i, n_span[0])]) + "\n"
            if i > 0:
                span_str = "\n" + span_str
            all_spans.append(span_str)
            idx_noised.append(0)
        i = n_span[1]
        all_spans.append('[bopref]' + "\n".join(n_span[2]) + '[eopref]')
        idx_noised.append(1)
    if i < len(data):
        span_str = "\n".join([data[j] for j in range(i, len(data))])
        if i > 0:
            span_str = "\n" + span_str
        all_spans.append(span_str)
        idx_noised.append(0)

    return all_spans, idx_noised


inject_dialogue_noise(dialogues, bos_token, speaker_mask_ratio=0.5,
                          text_infilling_ratio=0.04, turn_split_prob=0.5,
                          add_noise_prob=1/4, insert_prob=0.5, shuffling_ratio=0.5,
                          drop_utt_prob=0.5):
    all_noised_data = []
    all_idx_noised = []

    dialogues = [[line.strip() for line in re.split(r"\n|\r\n", dialogue) if line.strip() != ''] for dialogue in dialogues]
    all_utterances = []
    [all_utterances.extend(dialogue) for dialogue in dialogues]

    for data in dialogues:

        if random.random() < add_noise_prob:
            #ratio_noised_max = np.random.uniform(0.4, 0.5)
            ratio_noised_max = 0.65 #TODO
            avg_span_len = 2.5 / (1 - drop_utt_prob) #TODO
            nb_spans = int(np.round(len(data) * ratio_noised_max / (avg_span_len)))
            #print("ratio_noised_max", ratio_noised_max, "avg_span_len", avg_span_len, "nb_spans", nb_spans)
            noised_data, idx_noised = add_noise(data, all_utterances, nb_spans, avg_span_len, speaker_mask_ratio, text_infilling_ratio, turn_split_prob, insert_prob, shuffling_ratio, drop_utt_prob)
        else:
            noised_data = ["\n".join(data)]
            idx_noised = [0]
        noised_data[0] = bos_token + noised_data[0]

        #print("\n\n".join(data), "\n\n\n\n")
        #print("\n\n".join(noised_data), "\n\n\n\n")
        all_noised_data.extend(noised_data)
        all_idx_noised.extend(idx_noised)
    return all_noised_data, all_idx_noised


# choose infilling ratio (compromise between short and long masks versus few and many masks)