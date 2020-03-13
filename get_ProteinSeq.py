# coding: utf-8
# 将human.benchmark.list与protein2ipr.dat通过蛋白质ID映射，得到所需的human蛋白质的所有的Sequence信息


DATAROOT = "/ifs/gdata3/PredGO_data/"
protein_ids = []
all_domains = []

with open(DATAROOT + 'human.benchmark.list' ,'r') as f :
    for line in f:
        protein_ids.append(str(line.strip()))
assert len(protein_ids) == 13704
# print protein_ids

fasta = {}
num = 0
is_necessary = True
with open(DATAROOT + 'uniprot_sprot.fasta' ,'r') as f :
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            active_sequence_name = line[:]
            prot_id = line[4:10]
            if str(prot_id) not in protein_ids:
                is_necessary = False
                continue
            is_necessary = True
            if active_sequence_name not in fasta:
                fasta[active_sequence_name] = []
                num += 1
            continue
        if is_necessary == False:
            continue
        sequence = line
        fasta[active_sequence_name].append(sequence)

print len(fasta)

with open('data/humanProtSeq.fasta', 'w') as f:
    for key,var in fasta.items():
        f.write('{}\n'.format(str(key)))
        f.write('{}\n'.format('\n'.join([str(x) for x in var])))
