import numpy as np
import itertools
import collections
import os
import pickle
from scipy import sparse
import scipy.sparse as sp


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return list(adj_normalized.toarray())


def load_graph_data(fname, dis_threshold):
    with open(fname) as f:
        contents = f.readlines()
    ADJ_i = []
    ADJ_j = []
    ADJ_v = []
    num_locs = 0
    for line in contents:
        line = line.strip().split('\t')
        businessID1, businessID2, s_distance = int(line[0]), int(line[1]), float(line[2])
        num_locs = max(num_locs, businessID1, businessID2)
        if s_distance < dis_threshold:
            ADJ_i.append(businessID1)
            ADJ_j.append(businessID2)
            ADJ_v.append(1)
            ADJ_i.append(businessID2)
            ADJ_j.append(businessID1)
            ADJ_v.append(1)
    num_locs += 1
    ADJ = sparse.coo_matrix((ADJ_v, (ADJ_i, ADJ_j)), shape=(num_locs, num_locs))
    print 'loc num: ', num_locs
    return np.array(preprocess_adj(ADJ)), num_locs


def load_friends_data(fname):
    with open(fname) as f:
        contents = f.readlines()
    friends = {}
    ADJ_i, ADJ_j, ADJ_v = [], [], []
    num_users = 0
    for line in contents:
        line = line.strip().split()
        user1, user2 = int(line[0]), int(line[1])
        num_users = max(num_users, user1, user2)
        if user1 not in friends:
            friends[user1] = set()
        if user2 not in friends:
            friends[user2] = set()
        friends[user1].add(user2)
        friends[user2].add(user1)
    num_users += 1
    for user in friends:
        curr_friends = friends[user]
        for each_friend in curr_friends:
            ADJ_i.append(user)
            ADJ_j.append(each_friend)
            ADJ_v.append(1)
    ADJ = sparse.coo_matrix((ADJ_v, (ADJ_i, ADJ_j)), shape=(num_users, num_users))
    print 'users: ', num_users
    return np.array(preprocess_adj(ADJ)), num_users


def load_data(fname):
    # data = list(set(open(fname).readlines()))
    data = list(open(fname).readlines())
    data = [d.strip().split('\t') for d in data]
    ori_data_len = len(data)
    data = filter(lambda x: len(x) == 7, data)
    filter_data_len = len(data)
    if ori_data_len != filter_data_len:
        print("some rows got filtered")

    businessIds = [int(x[0]) for x in data]
    userIds = [int(x[1]) for x in data]
    geoScores = [[float(x[2])] for x in data]
    g_scores_1 = [[float(x[3])] for x in data]
    g_scores_2 = [[float(x[4])] for x in data]
    g_scores_3 = [[float(x[5])] for x in data]
    label = [int(x[-1]) for x in data]
    label = np.asarray(label)
    return businessIds, userIds, geoScores, g_scores_1, g_scores_2, g_scores_3, label


def load_checkin_data(fname):
    # data = list(set(open(fname).readlines()))
    data = list(open(fname).readlines())
    data = [d.strip().split('\t') for d in data]
    ori_data_len = len(data)
    data = filter(lambda x: len(x) == 7, data)
    filter_data_len = len(data)
    if ori_data_len != filter_data_len:
        print("some rows got filtered")

    businessIds = [int(x[0]) for x in data]
    userIds = [int(x[1]) for x in data]
    geoScores = [float(x[2]) for x in data]
    g_scores_1 = [float(x[3]) for x in data]
    g_scores_2 = [float(x[4]) for x in data]
    g_scores_3 = [float(x[5]) for x in data]
    label = [int(x[-1]) for x in data]
    label = np.asarray(label)

    user_num = 0
    for each_userID in userIds:
        user_num =max(user_num, each_userID)

    pos_train_dict = {}
    businessId_list = list()
    for idx in range(len(businessIds)):
        curr_businessId = businessIds[idx]
        businessId_list.append(curr_businessId)
        if curr_businessId not in pos_train_dict:
            pos_train_dict[curr_businessId] = []
        pos_train_dict[curr_businessId].append(
            [businessIds[idx], userIds[idx], geoScores[idx], g_scores_1[idx], g_scores_2[idx], g_scores_3[idx],
             label[idx]])
    dict_businessIds = collections.Counter(businessId_list)
    return pos_train_dict, dict_businessIds, user_num

def load_checkin_data_v1(fname_train, fname_valiate):
    # data = list(set(open(fname).readlines()))
    data_train = list(open(fname_train).readlines())
    data_validate = list(open(fname_valiate).readlines())
    data = data_train + data_validate
    data = [d.strip().split('\t') for d in data]
    ori_data_len = len(data)
    data = filter(lambda x: len(x) == 7, data)
    filter_data_len = len(data)
    if ori_data_len != filter_data_len:
        print("some rows got filtered")

    businessIds = [int(x[0]) for x in data]
    userIds = [int(x[1]) for x in data]
    geoScores = [float(x[2]) for x in data]
    g_scores_1 = [float(x[3]) for x in data]
    g_scores_2 = [float(x[4]) for x in data]
    g_scores_3 = [float(x[5]) for x in data]
    label = [int(x[-1]) for x in data]
    label = np.asarray(label)

    user_num = 0
    for each_userID in userIds:
        user_num =max(user_num, each_userID)

    pos_train_dict = {}
    businessId_list = list()
    for idx in range(len(businessIds)):
        curr_businessId = businessIds[idx]
        businessId_list.append(curr_businessId)
        if curr_businessId not in pos_train_dict:
            pos_train_dict[curr_businessId] = []
        pos_train_dict[curr_businessId].append(
            [businessIds[idx], userIds[idx], geoScores[idx], g_scores_1[idx], g_scores_2[idx], g_scores_3[idx],
             label[idx]])
    dict_businessIds = collections.Counter(businessId_list)
    return pos_train_dict, dict_businessIds, user_num
