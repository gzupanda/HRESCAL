import numpy as np
import os
import logging
import scipy.io as sio
import pandas as pd
import datetime

logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)
#logger.propagate = False
#Current path
cur_path = os.path.dirname(os.path.realpath( os.path.basename(__file__)))

class Triplets_set(object):
    """
    self.indexes attribute is a n*3 numpy array that are all triplets in the set,
    with their corresponding values (1 or -1) in self.values
    """

    def __init__(self, indexes, values):

        #Type cast to comply with theano types defined in downhill losses
        self.indexes = indexes.astype(np.int64)
        self.values = values.astype(np.float32)


class Experiment(object):

    def __init__(self, name, train, valid, test, positives_only=False, compute_ranking_scores=False, entities_dict=None,
                 relations_dict=None):
        """
        An experiment is defined by its train and test set, which are two Triplets_set objects.
        """

        self.name = name
        self.train = train
        self.valid = valid
        self.test = test
        self.train_tensor = None
        self.train_mask = None
        self.positives_only = positives_only
        self.entities_dict = entities_dict
        self.relations_dict = relations_dict

        if valid is not None:
            self.n_entities = len(np.unique(np.concatenate((train.indexes[:, 0], train.indexes[:, 2],
                                                            valid.indexes[:, 0], valid.indexes[:, 2],
                                                            test.indexes[:, 0], test.indexes[:, 2]))))
            self.n_relations = len(
                np.unique(np.concatenate((train.indexes[:, 1], valid.indexes[:, 1], test.indexes[:, 1]))))
        else:
            self.n_entities = len(np.unique(
                np.concatenate((train.indexes[:, 0], train.indexes[:, 2], test.indexes[:, 0], test.indexes[:, 2]))))
            self.n_relations = len(np.unique(np.concatenate((train.indexes[:, 1], test.indexes[:, 1]))))

        logger.info("Nb entities: " + str(self.n_entities))
        logger.info("Nb relations: " + str(self.n_relations))
        logger.info("Nb obs triples: " + str(train.indexes.shape[0]))

#parse datasets from text files
def parse_line(filename, line,i):
    line = line.strip().split("\t")
    sub = line[0]
    rel = line[1]
    obj = line[2]
    val = 1

    return sub,obj,rel,val

# constuction of adjacent tensors
def triples2tensor(data):
    (m,n) = data.shape
    k = len(data)
    tensor = np.array((m,n,k))

#--------------------get adjacent matrix of training data-----------------------------
def getAdjacentMatrix(tri,entities_i,name,folder):
    #get all triples(index)
    triples=[]
    for key in tri.keys():
        triples.append(key)
    len_triples = len(list(triples))
    # get all entities(index)
    enities = []
    for key in entities_i.keys():
        enities.append(key)
    len_entities = len(list(enities))
    print("The number of enities:",len_entities,"The number of triples:",len_triples)
    #construct a new 0 matrix
    fact_mat = np.zeros((len_entities,len_entities))
    #print(fact_mat)
    # construct adjacent matrix
    print("Starting to construct adjacent maxtrix:")
    """
    n =0
    for m in list(triples):
        if n < 100:
            print("m=",m)
            n+=1
        else:break
    """
    for k in list(triples):
        #print("k=",k)
        for i in range(len_triples):
            #print("i=", i)
            if i ==k[0]:
                for j in range((len_triples)):
                    #print("j=", j)
                    if j == k[2]:
                        fact_mat[i][j] = 1
                        print("INFO:",datetime.datetime.now(), "i=",i, "j=",j, "k=",k)
                    elif j<k[2]:
                        continue
                    else:
                        break
            elif  i<k[0]:
                continue
            else:break
    pd.DataFrame(fact_mat).to_csv(folder + name + '.csv')
    print("CSV is 2d!")
    fact_mat = fact_mat.reshape(len_entities,len_entities,1)
    sio.savemat(folder + name + '.mat',{'Rs':fact_mat})
    print("mat is 3d!")
    print("Constructing adjacent matrix successful! Its shape is ",fact_mat.shape)

def load_triples_from_txt(filenames, entities_indexes=None, relations_indexes=None, add_sameas_rel=False,
                          parse_line=parse_line):
    """
    Take a list of file names and build the corresponding dictionnary of triples
    """

    if entities_indexes is None:
        entities_indexes = dict()
        entities = set()
        next_ent = 0
    else:
        entities = set(entities_indexes)
        next_ent = max(entities_indexes.values()) + 1

    if relations_indexes is None:
        relations_indexes = dict()
        relations = set()
        next_rel = 0
    else:
        relations = set(relations_indexes)
        next_rel = max(relations_indexes.values()) + 1

    data = dict()
    #print(data.values())

    for filename in filenames:
        with open(filename) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):

            sub, obj, rel, val = parse_line(filename, line, i)

            if sub in entities:
                sub_ind = entities_indexes[sub]
            else:
                sub_ind = next_ent
                next_ent += 1
                entities_indexes[sub] = sub_ind
                entities.add(sub)

            if obj in entities:
                obj_ind = entities_indexes[obj]
            else:
                obj_ind = next_ent
                next_ent += 1
                entities_indexes[obj] = obj_ind
                entities.add(obj)

            if rel in relations:
                rel_ind = relations_indexes[rel]
            else:
                rel_ind = next_rel
                next_rel += 1
                relations_indexes[rel] = rel_ind
                relations.add(rel)

            data[(sub_ind, rel_ind, obj_ind)] = val

    if add_sameas_rel:
        rel = "sameAs_"
        rel_ind = next_rel
        next_rel += 1
        relations_indexes[rel] = rel_ind
        relations.add(rel)
        for sub in entities_indexes:
            for obj in entities_indexes:
                if sub == obj:
                    data[(entities_indexes[sub], rel_ind, entities_indexes[obj])] = 1
                else:
                    data[(entities_indexes[sub], rel_ind, entities_indexes[obj])] = -1

    return data, entities_indexes, relations_indexes

def build_data(name, path = 'E:/experiments/rescal-bilinear/rescal/dataprocessor/datasets/'):

    folder = path + '/' + name + '/'

    train_triples, entities_indexes, relations_indexes = load_triples_from_txt([folder + 'train.txt'],
                    add_sameas_rel = False, parse_line = parse_line)

    getAdjacentMatrix(train_triples, entities_indexes, "train", folder)
"""
    valid_triples, entities_indexes, relations_indexes =  load_triples_from_txt([folder + 'valid.txt'],
                    entities_indexes = entities_indexes , relations_indexes = relations_indexes,
                    add_sameas_rel = False, parse_line = parse_line)

    getAdjacentMatrix(valid_triples, entities_indexes, "valid", folder)

    test_triples, entities_indexes, relations_indexes =  load_triples_from_txt([folder + 'test.txt'],
                    entities_indexes = entities_indexes, relations_indexes = relations_indexes,
                    add_sameas_rel = False, parse_line = parse_line)

    getAdjacentMatrix(test_triples, entities_indexes, "test", folder)

    fb15k_triples, entities_indexes, relations_indexes = load_triples_from_txt([folder + 'fb15k.txt'],
                    entities_indexes=entities_indexes,relations_indexes=relations_indexes,
                    add_sameas_rel=False,parse_line=parse_line)

    getAdjacentMatrix(train_triples, entities_indexes, "fb15k", folder)

    train = Triplets_set(np.array(list(train_triples.keys())), np.array(list(train_triples.values())))
    valid = Triplets_set(np.array(list(valid_triples.keys())), np.array(list(valid_triples.values())))
    test = Triplets_set(np.array(list(test_triples.keys())), np.array(list(test_triples.values())))
"""
'''
    sio.savemat(folder + 'train.mat',train_triples)
    pd.DataFrame(train_triples,index = [0]).to_csv(folder + 'train.csv')

    sio.savemat(folder + 'valid.mat', valid_triples)
    pd.DataFrame(valid_triples, index=[0]).to_csv(folder + 'valid.csv')

    sio.savemat(folder + 'test.mat', test_triples)
    pd.DataFrame(test_triples, index=[0]).to_csv(folder + 'test.csv')

    return Experiment(name,train, valid, test, positives_only = True, compute_ranking_scores = True, entities_dict = entities_indexes, relations_dict = relations_indexes)
'''

if __name__ =="__main__":
    """
    This module is data preprocessing.It yields two kinds of data:mat format and csv format.
    """
    #parse data from .mat format
    fb15kexp = build_data(name='fb15k', path = cur_path + '/datasets/')
    #fb15k_237 = build_data(name='fb15k-237', path=cur_path + '/datasets/')
    #wn18 = build_data(name='wn18', path=cur_path + '/datasets/')
    #WN18RR = build_data(name='WN18RR', path=cur_path + '/datasets/')
    #parse data from .tsv format
    #nations = build_data(name='nations', path=cur_path + '/datasets/')
    #umls = build_data(name='umls', path=cur_path + '/datasets/')
    #umls = build_data(name='umls', path=cur_path + '/datasets/')