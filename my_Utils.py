# coding: utf-8
# some tools
# import pandas as pd
import os
import numpy as np
# from protVec.multi_k_model import MultiKModel

# 通过序列，获取该序列的k_mers矩阵
def mer_k(seq, protvec, k = 3, file_base = 'data/3_mers_base.csv'):
    three_mer = []
    with open(file_base, 'r') as f:
        for line in f:
            three_mer.append(str(line.strip()))
    protVec = protvec
    zeroVec = np.zeros(100, dtype= float).tolist()
    # zeroVec = np.zeros(8000, dtype=float).tolist()
    l = []
    seq_length = len(seq)
    for i in range(seq_length):
        t = seq[i:(i + k)]
        if (len(t)) == k:
            if t in three_mer:
                vec = protVec[t]
                l.append(vec)
            else:
                l.append(zeroVec)
            # break
    return l

# 通过序列，获取该序列的k_mers的Sentence
def mer_k_Sentence(seq, protdict, k = 3, file_base = 'data/3_mers_base.csv'):
    three_mer = []
    with open(file_base, 'r') as f:
        for line in f:
            three_mer.append(str(line.strip()))
    protDict = protdict
    # zeroVec = np.zeros(100, dtype= float).tolist()
    # zeroVec = np.zeros(8000, dtype=float).tolist()
    l = []
    seq_length = len(seq)
    for i in range(seq_length):
        t = seq[i:(i + k)]
        if (len(t)) == k:
            if t in three_mer:
                word = protDict[t]
                l.append(word)
            else:
                l.append(0)
            # break
    return l

# 通过domain，获取该protein的domain矩阵
def get_domain_matrix(domain_s, domainVec):
    l =[]
    for i in range(len(domain_s)):
        if domain_s[i] in domainVec:
            vec = domainVec[domain_s[i]]
            l.append(vec)
        else:
            # zero_vec = np.zeros((128), dtype=np.float).tolist()
            zero_vec = np.zeros((14242), dtype=np.float).tolist()
            l.append(zero_vec)
    return l

def make_k_mers_base():
    Amino_acid = ['A','C','D','E','F','G','H','I','K','L',
                  'M','N','P','Q','R','S','T','V','W','Y']
    k_mers = []
    for i in range(len(Amino_acid)):
        for j in range(len(Amino_acid)):
            for k in range(len(Amino_acid)):
                each_mer = Amino_acid[i] + Amino_acid[j] + Amino_acid[k]
                k_mers.append(each_mer)
    assert len(k_mers) == 8000
    # with open("data/3_mers_base.csv", 'w') as f:
    #     for line in range(len(k_mers)):
    #         f.write('{}\n'.format(str(k_mers[line])))
    return k_mers

def creat_kmer_metrix(k=3):
    kmers_base = make_k_mers_base()
    # zerolist = [0 for x in range(len(kmers_base))]
    kmers_dict = {}
    for i in range(len(kmers_base)):
        # temp = zerolist
        # temp[i] = 1
        kmers_dict[kmers_base[i]] = [0 for x in range(len(kmers_base))]
        kmers_dict[kmers_base[i]][i] = 1
    # print(kmers_dict['AAA'],kmers_dict['YYY'] )
    np.save('data/prot_onehot_dict.npy', kmers_dict)


def creat_kmer_Wordict(k=3):
    kmers_base = make_k_mers_base()
    # zerolist = [0 for x in range(len(kmers_base))]
    kmers_dict = {}
    for i in range(len(kmers_base)):
        # temp = zerolist
        # temp[i] = 1
        kmers_dict[kmers_base[i]] = i + 1
    # print(kmers_dict['AAA'],kmers_dict['YYY'] )
    np.save('data/prot_kmerWord_dict.npy', kmers_dict)
    with open('data/prot3mersWordict.csv', 'w') as f:
        for j in range(len(kmers_base)):
            f.write('{},'.format(kmers_base[j]))
            f.write('{}\n'.format(kmers_dict[kmers_base[j]]))

creat_kmer_Wordict()
# creat_kmer_metrix()
# print 'done'