# coding: utf-8
#主模型
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import time
import random
import my_Utils

# seqSet = 'data/seqSet.csv'
# domainSet = 'data/domainSet.csv'
# Benchmark_list = open('data.human.benchmark.list', 'r')
torch.manual_seed(100)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
time_start=time.time()
# GO_IDs = []


CFG = {
    'cfg00': [16, 'M', 16, 'M'],
    'cfg01': [16, 'M', 32, 'M'],
    'cfg02': [32, 'M'],
    'cfg03': [64, 'M'],
    'cfg04': [16, 'M', 16, 'M',32, 'M'],
    'cfg05': [64, 'M', 32, 'M',16, 'M'],
    'cfg06': [64, 'M', 32, 'M',32, 'M'],
    'cfg07': [128, 'M', 64, 'M2'],
    'cfg08': [512, 'M', 128, 'M2',32, 'M2'],
}
OUT_nodes = {
    'BP': 373,
    'MF': 171,
    'CC': 151,
}

Thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
              0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
              0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
              0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
              0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
              0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
              0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
              0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
              0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
              0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

ProtVec = np.load('data/protVec_dict.npy').item()
# ProtDict = np.load('data/prot_kmerWord_dict.npy').item()
Seqfile_name = 'data/yeast_seqSet.csv'
# Domainfile_name = 'data/domainSet.csv'
Domainfile_name = 'data/yeast_NewdomainSet.csv'
GOfile_name = 'data/yeast_ProteinGO.csv'


class Dataload(Dataset):
    def __init__(self, benchmark_list, seqfile_name, domainfile_name, GOfile_name, func='MF', transform=None):
        self.benchmark_list = benchmark_list
        self.sequeces = {}
        self.max_seq_len = 1500  # 序列长度小于5000的序列中的最大值wei 4981 大于1000的有1827条
        self.doamins = {}
        self.max_domains_len = 41  # 蛋白质所包含的domain数量的第二大值（最大值为1242，舍去该蛋白质）
        self.ppiVecs = {}
        self.GO_annotiations = {}
        # self.max_GOnums_len = 0     #含有GO标注最多的蛋白质的GO数量

        with open(seqfile_name, 'r') as f:  #seqfile_name = 'data/seqSet.csv'
            for line in f:
                items = line.strip().split(',')
                prot, seq = items[0], items[1]
                self.sequeces[prot] = seq
        self.protDict = ProtVec


        # ppi_file = 'PPI_data/selected_uniprot_protein_scores.csv'
        # ppi_file = 'PPI_data/selected_uniprot_protein_links.csv'
        ppi_file = 'PPI_data/selected_4932_protein_scores.csv'
        print(ppi_file)
        with open(ppi_file, 'r') as f:
            num = 1
            for line in f:
                if num == 1:
                    num = 2
                    continue
                items = line.strip().split(',')
                prot, vector =items[0], items[1:]
                self.ppiVecs[prot] = vector

        with open(GOfile_name, 'r') as f:       #GOfile_name = 'data/humanProteinGO.csv'
            num = 1
            for line in f:
                if num == 1:
                    num = 2
                    # items = line.strip().split(',')
                    # GO_IDs = items
                    continue
                items = line.strip().split(',')
                if func == 'BP':
                    prot, GO_annotiation = items[0], items[1:374]   #373
                elif func == 'MF':
                    prot, GO_annotiation = items[0], items[374:545] #171
                elif func == 'CC':
                    prot, GO_annotiation = items[0], items[545:]    #151
                # prot, GO_annotiation = items[0], items[1:]
                self.GO_annotiations[prot] = GO_annotiation

    def __getitem__(self, idx):
        iprID = self.benchmark_list[idx]

        # # 获取seq的输入向量
        # seq = self.sequeces[iprID]
        # if len(seq) >= self.max_seq_len:
        #     seq = seq[0:self.max_seq_len]
        # seqSentence = my_Utils.mer_k_Sentence(seq, self.protDict, 3)
        # seqSentence = np.array(seqSentence, dtype=int)
        # seqSentence = np.pad(seqSentence, (0, self.max_seq_len - len(seqSentence)), 'constant', constant_values=0)
        # seqSentence = torch.from_numpy(seqSentence).type(torch.LongTensor).cuda()

        # 获取seq的输入矩阵
        seq = self.sequeces[iprID]
        if len(seq) > self.max_seq_len:
            seq = seq[0:self.max_seq_len]
        seqMatrix = my_Utils.mer_k(seq, self.protDict, 3)
        seqMatrix = np.array(seqMatrix, dtype=float)
        if (seqMatrix.shape[0]) < self.max_seq_len:
            seqMatrix = np.pad(seqMatrix, ((0, self.max_seq_len - (seqMatrix.shape[0])), (0, 0)),
                               'constant', constant_values=0)
        seqMatrix = seqMatrix.T
        seqMatrix = torch.from_numpy(seqMatrix).type(torch.FloatTensor).cuda()

        domainStence = seqMatrix

        # 获取PPI的输入向量
        if iprID not in self.ppiVecs:
            ppiVect = np.zeros((6054), dtype=np.float).tolist()
        else:
            ppiVect = self.ppiVecs[iprID]
            ppiVect = [float(x) for x in ppiVect]
        ppiVect = torch.Tensor(ppiVect).cuda()
        ppiVect = ppiVect.type(torch.FloatTensor)

        #获取蛋白质的GO标注向量
        GO_annotiations = self.GO_annotiations[iprID]
        GO_annotiations = [int(x) for x in GO_annotiations]
        GO_annotiations = torch.Tensor(GO_annotiations).cuda()
        # GO_annotiations = GO_annotiations.type(torch.LongTensor)
        GO_annotiations = GO_annotiations.type(torch.FloatTensor)

        return seqMatrix, domainStence, ppiVect, GO_annotiations

    def __len__(self):
        return len(self.benchmark_list)     #返回蛋白质的数量


class weight_Dataload(Dataset):
    def __init__(self, benchmark_list, seqdict, ppidict, GOfile_name, func = 'MF', transform=None):
        self.benchmark = benchmark_list
        self.weghtdict = {}
        self.GO_annotiations = {}

        for i in range(len(benchmark_list)):
            prot = benchmark_list[i]
            # temp = torch.cat((seqdict[prot], domaindict[prot]), 0)
            # temp = torch.cat((temp, ppidict[prot]), 0)
            temp = [seqdict[prot], ppidict[prot]]
            temp = np.array(temp)
            self.weghtdict[benchmark_list[i]] = temp.flatten().tolist()
            assert len(seqdict[prot]) == len(ppidict[prot]) == OUT_nodes[func]

        with open(GOfile_name, 'r') as f:       #GOfile_name = 'data/humanProteinGO.csv'
            num = 1
            for line in f:
                if num == 1:
                    num = 2
                    # items = line.strip().split(',')
                    # GO_IDs = items
                    continue
                items = line.strip().split(',')
                if func == 'BP':
                    prot, GO_annotiation = items[0], items[1:374]   #373
                elif func == 'MF':
                    prot, GO_annotiation = items[0], items[374:545] #171
                elif func == 'CC':
                    prot, GO_annotiation = items[0], items[545:]    #151
                # prot, GO_annotiation = items[0], items[1:]
                self.GO_annotiations[prot] = GO_annotiation



    def __getitem__(self, idx):
        prot = self.benchmark[idx]

        #获取weight_classifier的输入向量
        weight_features = self.weghtdict[prot]
        weight_features = [float(x) for x in weight_features]
        weight_features = torch.Tensor(weight_features).cuda()
        weight_features = weight_features.type(torch.FloatTensor)


        # 获取蛋白质的GO标注向量
        GO_annotiations = self.GO_annotiations[prot]
        GO_annotiations = [int(x) for x in GO_annotiations]
        GO_annotiations = torch.Tensor(GO_annotiations).cuda()
        # GO_annotiations = GO_annotiations.type(torch.LongTensor)
        GO_annotiations = GO_annotiations.type(torch.FloatTensor)

        return weight_features, GO_annotiations

    def __len__(self):
        return len(self.benchmark)


class Seq_Module(nn.Module):
    def __init__(self, func):
        super(Seq_Module, self).__init__()
        # self.seq_emblayer = nn.Embedding(8001, 128, padding_idx=0)
        self.seq_CNN = self.SeqConv1d(CFG['cfg05']).cuda()
        self.seq_FClayer = nn.Linear(3008, 1024).cuda()
        self.seq_outlayer = nn.Linear(1024, OUT_nodes[func]).cuda()

    def forward(self, seqMatrix):
        # seqMatrix = self.seq_emblayer(seqSentence)
        seq_out = self.seq_CNN(seqMatrix)
        seq_out = seq_out.view(seq_out.size(0), -1)  # 展平多维的卷积图
        # print(seq_out)
        seq_out = F.dropout(self.seq_FClayer(seq_out), p=0.3, training=self.training)
        seq_out = F.relu(seq_out)
        seq_out = self.seq_outlayer(seq_out)
        seq_out = F.sigmoid(seq_out)
        return seq_out

    # sequence的1D_CNN模型
    def SeqConv1d(self, cfg):
        layers = []
        in_channels = 100
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2)]
            elif x == 'M2':
                layers += [nn.MaxPool1d(kernel_size=2, stride=1)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=16, stride=1, padding=8),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class PPI_Module(nn.Module):
    def __init__(self, func):
        super(PPI_Module, self).__init__()
        self.ppi_inputlayer = nn.Linear(6054, 2048).cuda()
        self.ppi_hiddenlayer = nn.Linear(2048, 1024).cuda()
        self.ppi_outlayer = nn.Linear(1024, OUT_nodes[func]).cuda()

    def forward(self, ppiVec):
        ppi_out = F.dropout(self.ppi_inputlayer(ppiVec), p=0.00005, training=self.training)
        ppi_out = F.dropout(self.ppi_hiddenlayer(ppi_out), p=0.3, training=self.training)
        ppi_out = self.ppi_outlayer(ppi_out)
        ppi_out = F.sigmoid(ppi_out)
        return ppi_out


class Weight_classifier(nn.Module):
    def __init__(self, func):
        super(Weight_classifier, self).__init__()
        # self.weight_layer = nn.Linear(OUT_nodes[func]*3, OUT_nodes[func])
        self.weight_layer = MaskedLinear(OUT_nodes[func]*2, OUT_nodes[func], 'data/yeast_{}_maskmatrix.csv'.format(func), func).cuda()
        self.outlayer = MaskedLinear(OUT_nodes[func], OUT_nodes[func], 'data/yeast_{}_maskmatrix_out.csv'.format(func), func).cuda()
        # self.outlayer= nn.Linear(OUT_nodes[func], OUT_nodes[func])

    def forward(self, weight_features):
        weight_out = self.weight_layer(weight_features)
        # weight_out = F.sigmoid(weight_out)
        weight_out = F.relu(weight_out)
        weight_out = F.sigmoid(self.outlayer(weight_out))
        return weight_out


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, relation_file, func, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

        mask = self.readRelationFromFile(relation_file, func)
        self.register_buffer('mask', mask)
        self.iter = 0

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

    def readRelationFromFile(self, relation_file, func):
        mask = []
        with open(relation_file, 'r') as f:
            for line in f:
                if relation_file == 'data/yeast_{}_maskmatrix_out.csv'.format(func):
                    l = [int(x) for x in line.strip().split(',')]
                else:
                    l = [int(x) for x in line.strip().split(',')[OUT_nodes[func]:]]
                    assert len(l) == OUT_nodes[func]*2
                for item in l:
                    assert item == 1 or item == 0  # relation 只能为0或者1
                mask.append(l)
        return Variable(torch.Tensor(mask))


def benchmark_set_split(term_arg='MF'):
    benchmark_file = 'data/yeast_{}_benchmarkSet.csv'.format(term_arg)
    print(benchmark_file)
    trainset, testset = [], []
    all_data = []
    with open(benchmark_file, 'r') as f:
        for line in f:
            item = line.strip()
            all_data.append(item)
    idx_list = np.arange(len(all_data)).tolist()
    # nums = {
    #     'BP': 10000,
    #     'MF': 10000,
    #     'CC': 10600,
    #     'test': 10
    # }
    nums = {
        'BP': 5000,
        'MF': 5100,
        'CC': 5200,
        'test': 30
    }
    # random_index = random.sample(idx_list, nums['test'])   #11000，在0--idx_list范围内随机产生nums[term_arg}个随机数
    #
    random_index = []
    with open('data/yeast_{}_random_index.csv'.format(term_arg), 'r') as f:
        for line in f:
            item = line.strip().split(',')
            random_index.append(int(item[0]))

    for i in range(len(all_data)):
        if i in random_index:
            trainset.append(all_data[i])
        else:
            testset.append(all_data[i])
    assert len(trainset) + len(testset) == len(all_data)
    # testset = trainset
    return trainset, testset


def calculate_performance(actual, pred_prob, threshold=0.4, average='micro'):
    pred_lable = []
    for l in range(len(pred_prob)):
        eachline = (np.array(pred_prob[l]) > threshold).astype(np.int)
        eachline = eachline.tolist()
        pred_lable.append(eachline)
    f_score = f1_score(np.array(actual), np.array(pred_lable), average=average)
    recall = recall_score(np.array(actual), np.array(pred_lable), average=average)
    # roc_auc = roc_auc_score(np.array(actual), np.array(pred_lable), average=average)
    precision = precision_score(np.array(actual), np.array(pred_lable), average=average)
    # fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred_lable).flatten(), pos_label=1)
    # auc_score = auc(fpr, tpr)
    return f_score, recall,  precision


def cacul_aupr(lables, pred):
    precision, recall, _thresholds = metrics.precision_recall_curve(lables, pred)
    aupr = metrics.auc(recall, precision)
    return aupr


def Seq_train(learningrate, batchsize, train_benchmark, test_benchmark, epochtime, func='MF'):
    print('{}  seqmodel start'.format(func))
    seq_model = Seq_Module(func).cuda()
    batch_size = batchsize
    learning_rate = learningrate
    epoch_times = epochtime
    print(seq_model)
    print('batch_size_{},learning_rate_{},epoch_times_{}'.format(batch_size, learning_rate, epoch_times))
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(seq_model.parameters(), lr=learning_rate, weight_decay = 0.00001)

    train_dataset = Dataload(train_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Dataload(test_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    seq_model.train()
    best_fscore = 0
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0
        for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(train_data_loader):
            seqMatrix = Variable(seqMatrix).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations)
            GO_annotiations = Variable(GO_annotiations).cuda()
            out = seq_model(seqMatrix)
            optimizer.zero_grad()
            loss = loss_function(out, GO_annotiations)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.data[0]
        epoch_loss = "{}".format(_loss / batch_num)
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []
        for idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
            seqMatrix = Variable(seqMatrix).cuda()
            GO_annotiations = Variable(GO_annotiations).cuda()
            out = seq_model(seqMatrix)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(GO_annotiations.data[0].cpu().tolist())
            one_loss = loss_function(out, GO_annotiations)
            t_loss += one_loss.data[0]
        test_loss = "{}".format(t_loss / test_batch_num)
        fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        score_dict = {}
        each_best_fcore = 0

        each_best_scores = []
        for i in range(len(Thresholds)):
            f_score, recall, precision = calculate_performance(
                actual, pred, threshold=Thresholds[i], average='micro')
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores
        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            torch.save(seq_model, 'savedpkl/Seq1DNVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times))
        t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score = each_best_scores[3], each_best_scores[4]
        print('epoch{},loss{},testloss:{},t{},f_score{}, auc{}, recall{}, precision{}'.format(
            epoch, epoch_loss, test_loss, t, f_score, auc_score, recall, precision))
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score = best_scores[3], best_scores[4]
    print('lr:{},batch:{},epoch{},f_max:{}\nauc{},recall_max{},prec_max{},threshold:{}'.format(
        learning_rate, batch_size, epoch_times,
        f_max, bestauc_score, recall_max, prec_max, bestthreshold))
    test_Seqmodel = torch.load('savedpkl/Seq1DNVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times)).cuda()
    t_loss = 0
    seq_test_outs = {}
    # seq_test_outs = []
    pred = []
    actual = []
    score_dict = {}
    batch_num = 0
    for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
        seqMatrix = Variable(seqMatrix).cuda()
        GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_Seqmodel(seqMatrix)
        batch_num += 1
        seq_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred.append(out.data[0].cpu().tolist())
        actual.append(GO_annotiations.data[0].cpu().tolist())
        loss = loss_function(out, GO_annotiations)
        t_loss += loss.data[0]
    test_loss = "{}".format(t_loss / batch_num)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    each_best_fcore = 0
    for i in range(len(Thresholds)):
        f_score, recall, precision = calculate_performance(
            actual, pred, threshold=Thresholds[i], average='micro')
        if f_score > each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
        scores = [f_score, recall, precision, auc_score]
        score_dict[Thresholds[i]] = scores
    bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    prec_max, bestauc_score = each_best_scores[3], each_best_scores[4]

    print('test_loss:{},lr:{},batch:{},epoch{},f_max:{}\nauc_score{},recall_max{},prec_max{},threshold:{}'.format(
        test_loss, learning_rate, batch_size, epoch_times,
        f_max, auc_score ,recall_max, prec_max, bestthreshold))

    # with open('out/weight_out/Seqout{}_lr{}_bat{}_epo{}.csv'.format(
    #         func, learning_rate, batch_size, epoch_times), 'w') as f:
    #     f.write('lr:{},batchsize:{},epochtimes:{}\n'.format(learning_rate, batch_size, epoch_times))
    #     f.write('f_max:{},recall_max{},prec_max{},auc_score:{}\n'.format(
    #         f_max,recall_max, prec_max, auc_score))
    #     f.write('threshold,f_score,recall,precision, roc_auc,auc\n')
    #     for i in range(len(Thresholds)):
    #         f.write('{},'.format(str(Thresholds[i])))
    #         f.write('{}\n'.format(','.join(str(x) for x in score_dict[Thresholds[i]])))
    #     for key, var in seq_test_outs.items():
    #         f.write('{},'.format(str(key)))
    #         f.write('{}\n'.format(','.join(str(x) for x in var)))

    #获取再最优模型下的训练集的输出
    train_out_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    seq_train_outs = {}
    for batch_idx, (seqMatrix, domainsMatrix, ppiVect, GO_annotiations) in enumerate(train_out_loader):
        seqMatrix = Variable(seqMatrix).cuda()
        # GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_Seqmodel(seqMatrix)
        seq_train_outs[train_benchmark[batch_idx]] = out.data[0].cpu().tolist()
    return seq_train_outs, seq_test_outs,bestthreshold        #返回再最优的Seq模型下的训练集的输出和测试集的输出，用于训练weight_classifier


def PPI_train(learningrate, batchsize, train_benchmark, test_benchmark, epochtime, func='MF'):
    print('{}  PPImodel start'.format(func))
    ppi_model = PPI_Module(func).cuda()
    batch_size = batchsize
    learning_rate = learningrate
    epoch_times = epochtime
    print(ppi_model)
    print('batch_size_{},learning_rate_{},epoch_times_{}'.format(batch_size, learning_rate, epoch_times))
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(ppi_model.parameters(), lr=learning_rate, weight_decay=0.00001)

    train_dataset = Dataload(train_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Dataload(test_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    ppi_model.train()
    best_fscore = 0
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0
        for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(train_data_loader):
            ppiVect = Variable(ppiVect).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations)
            GO_annotiations = Variable(GO_annotiations).cuda()
            out = ppi_model(ppiVect)
            optimizer.zero_grad()
            loss = loss_function(out, GO_annotiations)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.data[0]
        epoch_loss = "{}".format(_loss / batch_num)
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []
        for idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
            ppiVect = Variable(ppiVect).cuda()
            GO_annotiations = Variable(GO_annotiations).cuda()
            out = ppi_model(ppiVect)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(GO_annotiations.data[0].cpu().tolist())
            one_loss = loss_function(out, GO_annotiations)
            t_loss += one_loss.data[0]
        test_loss = "{}".format(t_loss / test_batch_num)
        fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        score_dict = {}
        each_best_fcore = 0
        each_best_scores = []
        for i in range(len(Thresholds)):
            f_score, recall, precision = calculate_performance(
                actual, pred, threshold=Thresholds[i], average='micro')
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores
        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            torch.save(ppi_model,
                       'savedpkl/PPINVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times))
        t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score = each_best_scores[3], each_best_scores[4]
        print('epoch{},loss{},testloss:{},t{},f_score{}, auc{}, recall{}, precision{}'.format(
            epoch, epoch_loss, test_loss, t, f_score, auc_score, recall, precision))
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score = best_scores[3], best_scores[4]
    print('lr:{},batch:{},epoch{},f_max:{}\nauc{},recall_max{},prec_max{},threshold:{}'.format(
        learning_rate, batch_size, epoch_times,
        f_max, bestauc_score, recall_max, prec_max, bestthreshold))
    test_PPImodel = torch.load(
        'savedpkl/PPINVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times)).cuda()
    t_loss = 0
    ppi_test_outs = {}
    pred = []
    actual = []
    score_dict = {}
    batch_num = 0
    for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
        ppiVect = Variable(ppiVect).cuda()
        GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_PPImodel(ppiVect)
        batch_num += 1
        ppi_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred.append(out.data[0].cpu().tolist())
        actual.append(GO_annotiations.data[0].cpu().tolist())
        loss = loss_function(out, GO_annotiations)
        t_loss += loss.data[0]
    test_loss = "{}".format(t_loss / batch_num)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    each_best_fcore = 0
    for i in range(len(Thresholds)):
        f_score, recall, precision = calculate_performance(
            actual, pred, threshold=Thresholds[i], average='micro')
        if f_score > each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
        scores = [f_score, recall, precision, auc_score]
        score_dict[Thresholds[i]] = scores
    bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    prec_max, bestauc_score = each_best_scores[3], each_best_scores[4]

    print('test_loss:{},lr:{},batch:{},epoch{},f_max:{}\nauc_score{},recall_max{},prec_max{},threshold:{}'.format(
        test_loss, learning_rate, batch_size, epoch_times,
        f_max, auc_score, recall_max, prec_max, bestthreshold))

    # with open('out/weight_out/PPIout{}_lr{}_bat{}_epo{}.csv'.format(
    #         func, learning_rate, batch_size, epoch_times), 'w') as f:
    #     f.write('lr:{},batchsize:{},epochtimes:{}\n'.format(learning_rate, batch_size, epoch_times))
    #     f.write('f_max:{},recall_max{},prec_max{},auc_score:{}\n'.format(
    #         f_max, recall_max, prec_max, auc_score))
    #     f.write('threshold,f_score,recall,precision, roc_auc,auc\n')
    #     for i in range(len(Thresholds)):
    #         f.write('{},'.format(str(Thresholds[i])))
    #         f.write('{}\n'.format(','.join(str(x) for x in score_dict[Thresholds[i]])))
    #     for key, var in ppi_test_outs.items():
    #         f.write('{},'.format(str(key)))
    #         f.write('{}\n'.format(','.join(str(x) for x in var)))

    # 获取再最优模型下的训练集的输出
    train_out_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    ppi_train_outs = {}
    for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(train_out_loader):
        ppiVect = Variable(ppiVect).cuda()
        # GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_PPImodel(ppiVect)
        ppi_train_outs[train_benchmark[batch_idx]] = out.data[0].cpu().tolist()
    return ppi_train_outs, ppi_test_outs, bestthreshold  # 返回再最优的PPI模型下的训练集的输出和测试集的输出，用于训练weight_classifier


# def Main(train_benchmark, test_benchmark, func='MF'):
def Main(func='MF'):
    train_benchmark, test_benchmark = benchmark_set_split(func)
    if func == 'BP':
        seq_train_out, seq_test_out, seq_t = Seq_train(0.0001, 16, train_benchmark, test_benchmark, 30, func)  # 15
    else:
        seq_train_out, seq_test_out, seq_t = Seq_train(0.001, 8, train_benchmark, test_benchmark, 17, func)  # 15
    ppi_train_out, ppi_test_out, ppi_t = PPI_train(0.0001, 8, train_benchmark, test_benchmark, 38, func)  # 40


    print('{}  Weight_model start'.format(func))
    weight_model = Weight_classifier(func).cuda()
    batch_size = 32
    learning_rate = 0.001
    epoch_times = 40
    print(weight_model)
    print('batch_size_{},learning_rate_{},epoch_times_{}'.format(batch_size, learning_rate, epoch_times))
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(weight_model.parameters(), lr=learning_rate, weight_decay=0.00001)

    train_dataset = weight_Dataload(train_benchmark, seq_train_out, ppi_train_out, GOfile_name, func=func)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = weight_Dataload(test_benchmark, seq_test_out, ppi_test_out, GOfile_name, func=func)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # weight_model.train()
    # best_fscore = 0
    # for epoch in range(epoch_times):
    #     _loss = 0
    #     batch_num = 0
    #     for batch_idx, (weight_features, GO_annotiations) in enumerate(train_data_loader):
    #         weight_features = Variable(weight_features).cuda()
    #         # print(weight_features)
    #         GO_annotiations = torch.squeeze(GO_annotiations)
    #         GO_annotiations = Variable(GO_annotiations).cuda()
    #         out = weight_model(weight_features)
    #         optimizer.zero_grad()
    #         loss = loss_function(out, GO_annotiations)
    #         batch_num += 1
    #         loss.backward()
    #         optimizer.step()
    #         _loss += loss.data[0]
    #     epoch_loss = "{}".format(_loss / batch_num)
    #     t_loss = 0
    #     test_batch_num = 0
    #     pred = []
    #     actual = []
    #     for idx, (weight_features, GO_annotiations) in enumerate(test_data_loader):
    #         weight_features = Variable(weight_features).cuda()
    #         GO_annotiations = Variable(GO_annotiations).cuda()
    #         out = weight_model(weight_features)
    #         test_batch_num = test_batch_num + 1
    #         pred.append(out.data[0].cpu().tolist())
    #         actual.append(GO_annotiations.data[0].cpu().tolist())
    #         one_loss = loss_function(out, GO_annotiations)
    #         t_loss += one_loss.data[0]
    #     test_loss = "{}".format(t_loss / test_batch_num)
    #     fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    #     auc_score = auc(fpr, tpr)
    #     score_dict = {}
    #     each_best_fcore = 0
    #     each_best_scores = []
    #     for i in range(len(Thresholds)):
    #         f_score, recall, precision = calculate_performance(
    #             actual, pred, threshold=Thresholds[i], average='micro')
    #         if f_score >= each_best_fcore:
    #             each_best_fcore = f_score
    #             each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
    #         scores = [f_score, recall, precision, auc_score]
    #         score_dict[Thresholds[i]] = scores
    #     if each_best_fcore >= best_fscore:
    #         best_fscore = each_best_fcore
    #         best_scores = each_best_scores
    #         best_score_dict = score_dict
    #         torch.save(weight_model,
    #                    'savedpkl/WeightNVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times))
    #     t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    #     precision, auc_score = each_best_scores[3], each_best_scores[4]
    #     print('epoch{},loss{},testloss:{},t{},f_score{}, auc{}, recall{}, precision{}'.format(
    #         epoch, epoch_loss, test_loss, t, f_score, auc_score, recall, precision))
    # bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    # prec_max, bestauc_score = best_scores[3], best_scores[4]
    # print('lr:{},batch:{},epoch{},f_max:{}\nauc{},recall_max{},prec_max{},threshold:{}'.format(
    #     learning_rate, batch_size, epoch_times,
    #     f_max, bestauc_score, recall_max, prec_max, bestthreshold))
    # return best_scores

    test_weight_model = torch.load(
        'savedpkl/WeightNVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times)).cuda()
    t_loss = 0
    weight_test_outs = {}
    pred = []
    actual = []
    score_dict = {}
    batch_num = 0
    for batch_idx, (weight_features, GO_annotiations) in enumerate(test_data_loader):
        weight_features = Variable(weight_features).cuda()
        GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_weight_model(weight_features)
        batch_num += 1
        weight_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred.append(out.data[0].cpu().tolist())
        actual.append(GO_annotiations.data[0].cpu().tolist())
        loss = loss_function(out, GO_annotiations)
        t_loss += loss.data[0]
    test_loss = "{}".format(t_loss / batch_num)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    aupr = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
    each_best_fcore = 0
    for i in range(len(Thresholds)):
        f_score, recall, precision = calculate_performance(
            actual, pred, threshold=Thresholds[i], average='micro')
        if f_score > each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score, aupr]
        scores = [f_score, recall, precision, auc_score]
        score_dict[Thresholds[i]] = scores
    bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    prec_max, bestauc_score, aupr_score = each_best_scores[3], each_best_scores[4], each_best_scores[5]

    print('test_loss:{},lr:{},batch:{},epoch{},f_max:{}\nauc_score{},recall_max{},prec_max{},threshold:{}'.format(
        test_loss, learning_rate, batch_size, epoch_times,
        f_max, auc_score, recall_max, prec_max, bestthreshold))

    with open('out/weight_out/Weight_nodom_out{}_lr{}_bat{}_epo{}.csv'.format(
            func, learning_rate, batch_size, epoch_times), 'w') as f:
        f.write('lr:{},batchsize:{},epochtimes:{}\n'.format(learning_rate, batch_size, epoch_times))
        f.write('f_max:{},recall_max{},prec_max{},auc_score:{}\n'.format(
            f_max, recall_max, prec_max, auc_score))
        f.write('threshold,f_score,recall,precision, roc_auc,auc\n')
        for i in range(len(Thresholds)):
            f.write('{},'.format(str(Thresholds[i])))
            f.write('{}\n'.format(','.join(str(x) for x in score_dict[Thresholds[i]])))
        for key, var in weight_test_outs.items():
            f.write('{},'.format(str(key)))
            f.write('{}\n'.format(','.join(str(x) for x in var)))
    # pass
    # return each_best_scores


def read_benchmark(term_arg='MF'):
    benchmark_file = 'data/yeast_{}_benchmarkSet.csv'.format(term_arg)
    print(benchmark_file)
    all_data = []
    with open(benchmark_file, 'r') as f:
        for line in f:
            item = line.strip()
            all_data.append(item)
    return all_data


def validation(func='MF', k_fold=5):
    kf = KFold(n_splits=k_fold)
    benchmark = np.array(read_benchmark(func))
    scores = []
    for train_index, test_index in kf.split(benchmark):
        train_set = benchmark[train_index].tolist()
        test_set = benchmark[test_index].tolist()
        each_fold_scores = Main(train_set, test_set, func=func)
        scores.append(each_fold_scores)
    f_maxs, pre_maxs, rec_maxs, auc_s, aupr_s = [], [], [], [], []
    for i in range(len(scores)):
        f_maxs.append(scores[i][1])
        rec_maxs.append(scores[i][2])
        pre_maxs.append(scores[i][3])
        auc_s.append(scores[i][4])
        aupr_s.append(scores[i][5])
    f_mean = np.mean(np.array(f_maxs))
    rec_mean = np.mean(np.array(rec_maxs))
    pre_mean = np.mean(np.array(pre_maxs))
    auc_mean = np.mean(np.array(auc_s))
    aupr_mean = np.mean(np.array(aupr_s))
    print('{}:f_mean{},rec_mean{},pre_mean{},auc_mean{}, aupr_mean{}'.format(
        func, f_mean, rec_mean, pre_mean, auc_mean, aupr_mean))


# def get_out(test_benchmark, func):
#     test_dataset = Dataload(test_benchmark, ind_Seqfile_name, ind_Domainfile_name, ind_GOfile_name, func=func)
#     test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
#     # seq_model = torch.load('savedpkl/ind_Seq1DM_{}_8_0.001_15.pkl'.format(func)).cuda()
#     # domainmodel = torch.load('savedpkl/ind_Doamin1DM_{}_16_0.001_40.pkl'.format(func)).cuda()
#     # ppimodel = torch.load('savedpkl/ind_PPIM_{}_8_0.0001_40.pkl'.format(func)).cuda()
#     if func=='BP':
#         seq_model = torch.load('savedpkl/Seq1DM_{}_16_0.0001_30.pkl'.format(func)).cuda()
#     else:
#         seq_model = torch.load('savedpkl/Seq1DM_{}_8_0.001_17.pkl'.format(func)).cuda()
#     # seq_model = torch.load('savedpkl/Seq1DM_{}_8_0.001_17.pkl'.format(func)).cuda()
#     # domainmodel = torch.load('savedpkl/Doamin1DM_{}_16_0.001_38.pkl'.format(func)).cuda()
#     ppimodel = torch.load('savedpkl/PPIM_{}_8_0.0001_38.pkl'.format(func)).cuda()
#     weightmodel = torch.load('savedpkl/WeightNodomM_{}_32_0.001_40.pkl'.format(func)).cuda()
#     seq_test_dicts, doamin_test_dicts, ppi_test_dicts = {}, {}, {}
#     test_lable_dicts = {}
#
#     for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
#         seqMatrix = Variable(seqMatrix).cuda()
#         domainStence = Variable(domainStence).cuda()
#         ppiVect = Variable(ppiVect).cuda()
#         GO_annotiations = Variable(GO_annotiations).cuda()
#         seq_test_dicts[test_benchmark[batch_idx]] = seq_model(seqMatrix).data[0].cpu().tolist()
#         ppi_test_dicts[test_benchmark[batch_idx]] = ppimodel(ppiVect).data[0].cpu().tolist()
#
#
#     eval_dataset = weight_Dataload(test_benchmark, seq_test_dicts, ppi_test_dicts, ind_GOfile_name, func=func)
#     eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
#     weight_test_outs = {}
#     batch_num = 0
#     pred = []
#     actual = []
#     score_dict = {}
#     for batch_idx, (weight_features, GO_annotiations) in enumerate(eval_data_loader):
#         weight_features = Variable(weight_features).cuda()
#         # GO_annotiations = torch.squeeze(GO_annotiations)
#         GO_annotiations = Variable(GO_annotiations).cuda()
#         # print(GO_annotiations)
#         weight_out = weightmodel(weight_features)
#         batch_num += 1
#         weight_test_outs[test_benchmark[batch_idx]] = weight_out.data[0]
#         pred.append(weight_out.data[0].cpu().tolist())
#         actual.append(GO_annotiations.data[0].cpu().tolist())
#     fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
#     auc_score = auc(fpr, tpr)
#     each_best_fcore = 0
#     aupr = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
#     for i in range(len(Thresholds)):
#         f_score, recall, precision = calculate_performance(
#             actual, pred, threshold=Thresholds[i], average='micro')
#         if f_score > each_best_fcore:
#             each_best_fcore = f_score
#             each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
#         scores = [f_score, recall, precision, auc_score]
#         score_dict[Thresholds[i]] = scores
#     bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
#     prec_max, bestauc_score = each_best_scores[3], each_best_scores[4]
#
#     print('{}:f_max:{},auc_score{},recall_max{},prec_max{},aupr{}, threshold:{}\n'.format(
#         func, f_max, auc_score, recall_max, prec_max, aupr, bestthreshold))
#     # calculate_each_performance(actual, pred, func)



def read_bench(file):
    bench = []
    with open(file, 'r') as f:
        for line in f:
            item = line.strip()
            bench.append(item)
    return bench


# def weight_analysis(func='MF'):
#     # train_benchmark, test_benchmark = benchmark_set_split(func)
#     train_benchmark = read_bench('data/{}_benchmarkSet_2.csv'.format(func))
#     # il = [0.1, 0.13, 0.2, 0.3, 0.99]  # 0.1  bp
#     # il = [0.2, 0.3, 0.99,]    #0.38,0.33  mf
#     # il = [0.33, 0.4, 0.5, 0.99]   #0.29, 0.2
#
#
#
#     test_benchmark = read_bench('data/independent_data/{}_benchmarkSet_ind_final.csv'.format(func))
#     #
#     get_out(test_benchmark, func)
#     # for i in range(len(il)):
#     #
#     #     test_benchmark = read_bench('data/independent_data/{}_benchmarkSet_ind_{}.csv'.format(func, il[i]))
#     #
#     #
#     #     get_out(test_benchmark, func)


if __name__ == '__main__':
    Terms = ['BP', 'MF', 'CC']
    # validation(Terms[2], k_fold=5)
    Main('BP')
    Main('MF')
    Main('CC')
    # run(func=Terms[1])
    # learning_rates = [0.001]
    # # learning_rates = [0.001, 0.0001, 0.01, 0.00001]
    # batchsizes = [64, 32, 8, 16]
    # # batchsizes = [32]
    # is_first = False
    # for i in range(len(learning_rates)):
    #     for j in range(len(batchsizes)):
    #         if is_first:
    #             Main(learning_rates[i], batchsizes[j], 40, func=Terms[1], is_first=True)
    #             is_first = False
    #         else:
    #             Main(learning_rates[i], batchsizes[j], 40, func=Terms[2], is_first=False)
    # weight_analysis(Terms[0])
    # weight_analysis(Terms[1])
    # weight_analysis(Terms[2])
    time_end = time.time()

    print('time cost', time_end - time_start,'s')
    # os.system("/ifs/share/bin/Python2/bin/python new_ind_eval.py")

