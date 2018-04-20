import numpy as np
import os
import random
import csv
import sys

def delex(input_sentence, target_sentence, slots): # Delexicalization function

    ### INPUT SENTENCE KEY/VALUE DICTIONARY ###
    sentence_dict = {}
    for categories in input_sentence.split(','):
        pair = 0
        for element in categories.split('['):
            if pair % 2 == 0:
                key = element.strip()
                ('key:', key)
            else:
                value = element[:-1]
                ('value', value)
                sentence_dict[key] = value
            pair +=1
            (pair)

    ### REBUILDING INPUT SENTENCE THROUGH DELEXICALIZATION ###
    new_input  = ''
    for categories in input_sentence.split(','):
        pair = 0
        delex = False
        for element in categories.split('['):
            if pair % 2 == 0:
                new_input_base = []
                key = element.strip()
                ('key0', key)
                if key in slots:
                    delex = True
                new_input_base.append(key)
            else:
                value = element[:-1]
                ('value', value)
                if delex == True:
                    new_input_base.append('x-'+key)
                else:
                    new_input_base.append(value)
            pair+=1
        new_input += ' '.join(new_input_base)
        new_input += ', '

    ### REBUILDING TARGET SENTENCE THROUGH DELEXICALIZATION ###
    for key in slots:
        if key in sentence_dict:
            if (key == 'area') and (sentence_dict[key] == 'riverside'):
                target_sentence = target_sentence.replace('Riverside', 'x-'+key)
            else:
                target_sentence = target_sentence.replace(sentence_dict[key], 'x-'+key)

    return sentence_dict, new_input, target_sentence


def relex(sent_dic, target_text): # Relexicalization function
    default_dic = {"name": 'The Waterman',
                   "near": 'The Ranch',
                   "area": 'the Riverside', 
                   "eatType": 'restaurant',
                   "priceRange": 'average',
                   "familyFriendly" : 'not family friendly', 
                   "food" : 'French', 
                   "customer rating": '3 out of 5'}
    (sent_dic)
    for word in target_text.split():
        (word)
        if word[:2] == 'x-':
            key = word[2:]
            if (word[-1]=='.') or (word[-1]==','):
                key = key[:-1]
                (key)
            if key != 'customer':
                if key in sent_dic:
                    target_text = target_text.replace(word, sent_dic[key])
                elif (key == 'near') and ('area' in sent_dic):
                    target_text = target_text.replace(word, sent_dic['area'])
                elif (key == 'area') and ('near' in sent_dic):
                    target_text = target_text.replace(word, sent_dic['near'])
                elif key in default_dic:
                    target_text = target_text.replace(word, default_dic[key])
            elif (key == 'customer') and ('customer rating' in sent_dic):
                target_text = target_text.replace('x-customer rating', sent_dic['customer rating'])
            else:
                target_text = target_text.replace('x-customer rating', default_dic['customer rating'])
    return target_text
