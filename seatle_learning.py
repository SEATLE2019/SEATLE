import argparse
import matplotlib as mpl
import StringIO
import operator
import os

from meta_learning import *
from data_utils import load_graph_data, load_checkin_data, load_checkin_data_v1

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='Cha', help='city')

parser.add_argument('--mode', type=str, default='train', help='train or infer or visual')

parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--keep_rate', type=float, default=0.8, help='dropout keep rate')  # not used
parser.add_argument('--margin', type=float, default=0.1, help='hinge loss margin')

parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--n_chosen_business', type=int, default=50, help='number of business in each episode')
parser.add_argument('--n_reference', type=int, default=4, help='number of reference for each business')
parser.add_argument('--num_query_per_cls', type=int, default=2, help='number of query instances for each business')

parser.add_argument('--feature_dim', type=int, default=10, help='the feature dimension for a tuple after MLP')
parser.add_argument('--user_feature_dim', type=int, default=10, help='the feature dimension to represent a user')
parser.add_argument('--loc_feature_dim', type=int, default=10, help='the feature dimension to represent a business')
parser.add_argument('--context_vector_size', type=int, default=10,
                    help='the size of the context vector in the reference representative attention mechanism')

parser.add_argument('--num_heads', type=int, default=1, help='number of heads in the multihead attention')

parser.add_argument('--distance_threshold', type=float, default=0.01, help='the distance threshold in GCN')

parser.add_argument('--obv', type=int, default=4, help='Every obv we output the performance on training set')
parser.add_argument('--eval_epoch_num', type=int, default=1, help='evaluate after a certain epochs')
args = parser.parse_args()


def train():
    print('Loading and creating location graph data...')
    location_distance_file = 'data/' + args.city + '/data/' + args.city + '_businessGeoDistance.txt'
    location_adjacent_matrix, loc_num = load_graph_data(location_distance_file, args.distance_threshold)

    print('Loading checkin data...')
    train_pos_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_training_pos.txt'
    train_neg_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_training_neg.txt'
    vali_pos_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_validation_pos.txt'
    vali_neg_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_validation_neg.txt'
    test_pos_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_test_pos.txt'
    test_neg_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_test_neg.txt'

    # pos_train_dict, pos_train_businessIds_dict, user_num = load_checkin_data(train_pos_fname)
    # neg_train_dict, neg_train_businessIds_dict, _ = load_checkin_data(train_neg_fname)

    pos_train_dict, pos_train_businessIds_dict, user_num = load_checkin_data_v1(train_pos_fname, vali_pos_fname)
    neg_train_dict, neg_train_businessIds_dict, _ = load_checkin_data_v1(train_neg_fname, vali_neg_fname)

    pos_vali_dict, pos_vali_businessIds_dict, _ = load_checkin_data(test_pos_fname)
    neg_vali_dict, neg_vali_businessIds_dict, _ = load_checkin_data(test_neg_fname)

    pos_test_dict, pos_test_businessIds_dict, _ = load_checkin_data(test_pos_fname)
    neg_test_dict, neg_test_businessIds_dict, _ = load_checkin_data(test_neg_fname)

    nb_classes = len(pos_train_dict)  # nb_classes is equal to the number of businesses
    n_reference = args.n_reference
    n_chosen_business = min(args.n_chosen_business, nb_classes);  # Define  the number of classes in each episode
    n_episodes = int(nb_classes / n_chosen_business)

    n_query = min(args.num_query_per_cls,
                  int(np.min(list(pos_train_businessIds_dict.values()))) - 2)

    alg = MetaLearningWrapper(learning_rate=args.learning_rate, keep_rate=args.keep_rate, margin=args.margin,
                              obv=args.obv, num_query_per_cls=args.num_query_per_cls,
                              num_ref_per_cls=args.n_reference, user_feature_dim=args.user_feature_dim,
                              loc_feature_dim=args.loc_feature_dim, feature_dim=args.feature_dim,
                              context_vector_size = args.context_vector_size, num_heads=args.num_heads,
                              loc_ADJ=location_adjacent_matrix, loc_num=loc_num, user_num=user_num)
    alg.create(training=True)

    print 'Max user visits for a business in train: ', np.max(pos_train_businessIds_dict.values())
    print 'Min user visits for a business in train: ', np.min(pos_train_businessIds_dict.values())
    print 'Mean user visits for a business in train', np.mean(pos_train_businessIds_dict.values())
    print 'Number of queries for each business is: ', n_query

    business_ins_num = {}
    for key in pos_train_dict:
        business_ins_num[key] = len(pos_train_dict[key])

    sorted_businessIds = sorted(business_ins_num.items(), key=operator.itemgetter(1), reverse=True)
    businessId_list = [item[0] for item in sorted_businessIds]
    mean_visits = np.mean(pos_train_businessIds_dict.values())
    # by doing this, we increase the sampling probs for businesses with less check-ins
    businessId_prob_list = [item[1] + mean_visits for item in sorted_businessIds]
    total_visits = sum(businessId_prob_list)
    businessId_prob_list = [float(item) / total_visits for item in businessId_prob_list]

    max_vali_acc, max_test_acc = 0.0, 0.0
    for each_epoch in range(args.n_epochs):
        n_supports_list = [n_query] * n_episodes
        class_start = 0
        # businessId_list = np.random.permutation(nb_classes)
        for each_episode in range(n_episodes):
            curr_support = n_supports_list[each_episode]
            epi_classes = businessId_list[class_start:class_start + n_chosen_business]  # chosen classes
            class_start += n_chosen_business

            # Sample classes based on the number of their check-ins
            epi_classes = np.random.choice(businessId_list, n_chosen_business, p=businessId_prob_list, replace=True)

            reference_vec = [None for _ in range(n_chosen_business)]
            pos_vec = [None for _ in range(n_chosen_business)]
            neg_vec = [None for _ in range(n_chosen_business)]

            for i, epi_cls in enumerate(epi_classes):  # for each chosen class
                n_examples_pos = len(pos_train_dict[epi_cls])
                n_examples_neg = len(neg_train_dict[epi_cls])
                if n_examples_pos != n_examples_neg:
                    print 'Train: pos and neg instances number is different for businessId: ', epi_cls
                if min(n_examples_pos,
                       n_examples_neg) < curr_support:  # potential issue: business has different number of pos/neg
                    curr_support = min(n_examples_pos, n_examples_neg) - 1
                    print 'support got changed!'
                if n_reference + curr_support <= n_examples_pos:  # we have enough instances
                    pos_selected = np.random.permutation(n_examples_pos)[:n_reference + curr_support]
                else:  # we do not have enough instances
                    pos_selected = np.random.choice(n_examples_pos, n_reference + curr_support)

                ###########################
                # pos_selected = range(0, n_examples_pos)[:n_reference + curr_support]
                # pos_selected = np.asarray(pos_selected)
                ##########################
                reference_vec[i] = [pos_train_dict[epi_cls][_] for _ in pos_selected[:n_reference]]
                pos_vec[i] = [pos_train_dict[epi_cls][_] for _ in pos_selected[n_reference:]]

                if curr_support <= n_examples_neg:  # we have enough instances
                    neg_selected = np.random.permutation(n_examples_neg)[:curr_support]
                else:  # we do not have enough instances
                    neg_selected = np.random.choice(n_examples_neg, curr_support)

                neg_vec[i] = [neg_train_dict[epi_cls][_] for _ in neg_selected]

            alg.updateParameters(reference_vec, pos_vec, neg_vec, eval=True)

        # Evaluate the performance on validation and test after each epoch
        if each_epoch % args.eval_epoch_num == 0:
            total_vali_loss, total_vali_acc = 0.0, 0.0
            num_evaluate = 0
            for epi_cls in range(nb_classes):
                n_pos_examples_train = len(pos_train_dict[epi_cls])
                reference_ids = np.random.permutation(n_pos_examples_train)

                #######################
                # reference_ids = np.asarray(range(0, n_pos_examples_train))
                ########################
                reference_vec = [[pos_train_dict[epi_cls][_] for _ in reference_ids[:n_reference]]]

                n_pos_examples = len(pos_vali_dict[epi_cls])
                n_neg_examples = len(neg_vali_dict[epi_cls])
                assert n_pos_examples == n_neg_examples
                if n_pos_examples != n_neg_examples:
                    print 'Validation: pos and neg instances number is different for businessId: ', epi_cls

                vali_loss_cls, vali_acc_cls = 0.0, 0.0
                evaluate_num = n_pos_examples / args.num_query_per_cls
                query_start_idx = 0
                for evaludate_idx in range(evaluate_num):
                    pos_vec = [pos_vali_dict[epi_cls][query_start_idx:query_start_idx + args.num_query_per_cls]]
                    neg_vec = [neg_vali_dict[epi_cls][query_start_idx:query_start_idx + args.num_query_per_cls]]
                    query_start_idx += args.num_query_per_cls
                    vali_loss, vali_acc, _, _ = alg.evaluate(reference_vec, pos_vec, neg_vec, shuffle=True)
                    vali_loss_cls += vali_loss
                    vali_acc_cls += vali_acc

                if evaluate_num > 0:
                    total_vali_loss += (vali_loss_cls / evaluate_num)
                    total_vali_acc += (vali_acc_cls / evaluate_num)
                    num_evaluate += 1

            print ''
            curr_vali_acc = round(total_vali_acc / num_evaluate, 5)
            if curr_vali_acc > max_vali_acc:
                max_vali_acc = curr_vali_acc
                alg.saveModel('Model/', each_epoch)
                print('Optimized in validation set and checkpoint saved')
            print('Epoch: ', each_epoch, ' loss: ', round(total_vali_loss / num_evaluate, 7),
                  ' acc: ', curr_vali_acc,
                  ' max acc:', round(max_vali_acc, 7), ' [Vali]')

            total_test_loss, total_test_acc = 0.0, 0.0
            num_evaluate = 0
            for epi_cls in range(nb_classes):
                n_pos_examples_train = len(pos_train_dict[epi_cls])
                reference_ids = np.random.permutation(n_pos_examples_train)

                #######################
                # reference_ids = np.asarray(range(0, n_pos_examples_train))
                ########################
                reference_vec = [[pos_train_dict[epi_cls][_] for _ in reference_ids[:n_reference]]]

                n_pos_examples = len(pos_test_dict[epi_cls])
                n_neg_examples = len(neg_test_dict[epi_cls])
                if n_pos_examples != n_neg_examples:
                    print 'Test: pos and neg instances number is different for businessId: ', epi_cls

                test_loss_cls, test_acc_cls = 0.0, 0.0
                evaluate_num = n_pos_examples / args.num_query_per_cls
                query_start_idx = 0
                for evaludate_idx in range(evaluate_num):
                    pos_vec = [pos_test_dict[epi_cls][query_start_idx:query_start_idx + args.num_query_per_cls]]
                    neg_vec = [neg_test_dict[epi_cls][query_start_idx:query_start_idx + args.num_query_per_cls]]
                    query_start_idx += args.num_query_per_cls
                    test_loss, test_acc, _, _ = alg.evaluate(reference_vec, pos_vec, neg_vec, shuffle=True)
                    test_loss_cls += test_loss
                    test_acc_cls += test_acc

                if evaluate_num > 0:
                    total_test_loss += (test_loss_cls / evaluate_num)
                    total_test_acc += (test_acc_cls / evaluate_num)
                    num_evaluate += 1

            curr_test_acc = round(total_test_acc / num_evaluate, 5)
            if curr_test_acc > max_test_acc:
                max_test_acc = curr_test_acc
            print('Epoch: ', each_epoch, ' loss: ', round(total_test_loss / num_evaluate, 7),
                  ' acc: ', curr_test_acc,
                  ' max acc:', round(max_test_acc, 7), ' [Test]')
            print ''



def infer():
    print('Loading and creating location graph data...')
    location_distance_file = 'data/' + args.city + '/data/' + args.city + '_businessGeoDistance.txt'
    location_adjacent_matrix, loc_num = load_graph_data(location_distance_file, args.distance_threshold)

    print('Loading checkin data...')
    train_pos_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_training_pos.txt'
    vali_pos_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_validation_pos.txt'
    # train_pos_fname = 'data/' + args.city + '/data/'  + args.city + '_checkin_test_pos.txt'
    # vali_pos_fname = 'data/' + args.city + '/data/'  + args.city + '_checkin_test_pos.txt'
    eval_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_eval.txt'

    # pos_train_dict, pos_train_businessIds_dict, user_num = load_checkin_data(train_pos_fname)
    pos_train_dict, pos_train_businessIds_dict, user_num = load_checkin_data_v1(train_pos_fname, vali_pos_fname)
    eval_dict, eval_businessIds_dict, _ = load_checkin_data(eval_fname)

    nb_classes = len(pos_train_dict)  # it is equal to the number of busineses
    n_reference = args.n_reference
    n_users = len(eval_dict[0])
    print 'number of users: ', n_users

    alg = MetaLearningWrapper(learning_rate=args.learning_rate, keep_rate=1.0, margin=args.margin,
                              obv=args.obv, num_query_per_cls=n_users,
                              num_ref_per_cls=n_reference, user_feature_dim=args.user_feature_dim,
                              context_vector_size = args.context_vector_size, num_heads=args.num_heads,
                              loc_feature_dim=args.loc_feature_dim, feature_dim=args.feature_dim,
                              loc_ADJ=location_adjacent_matrix, loc_num=loc_num, user_num=user_num)
    alg.create(training=False)

    Str_IOString = StringIO.StringIO()
    for epi_cls in range(nb_classes):
        pos_vec = [None]
        pos_vec[0] = eval_dict[epi_cls]
        n_pos_examples_train = len(pos_train_dict[epi_cls])
        reference_ids = np.random.permutation(n_pos_examples_train)

        ######################
        # reference_ids = np.asarray(range(0, n_pos_examples_train))
        #####################
        reference_vec = [[pos_train_dict[epi_cls][_] for _ in reference_ids[:n_reference]]]

        vali_loss, vali_acc, pos_scores, neg_scores = alg.evaluate(reference_vec, pos_vec, pos_vec)
        for each_score in pos_scores:
            Str_IOString.write(str(each_score) + '\n')
        if epi_cls % 1000 == 0 and epi_cls > 0:
            print epi_cls, ' classes have been infered.'

    # Output the predicted scores into the file.
    out_fname = 'data/' + args.city + '/data/' + args.city + '_prediction_probs.txt'
    fout = open(out_fname, 'w')
    fout.write(Str_IOString.getvalue())
    fout.close()


def visualize():
    print('Loading and creating location graph data...')
    location_distance_file = 'data/' + args.city + '/data/' + args.city + '_businessGeoDistance.txt'
    location_adjacent_matrix, loc_num = load_graph_data(location_distance_file, args.distance_threshold)

    print('Loading checkin data...')
    train_pos_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_training_pos.txt'
    train_neg_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_training_neg.txt'
    vali_pos_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_validation_pos.txt'
    vali_neg_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_validation_neg.txt'
    test_pos_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_test_pos.txt'
    test_neg_fname = 'data/' + args.city + '/data/' + args.city + '_checkin_test_neg.txt'

    # pos_train_dict, pos_train_businessIds_dict, user_num = load_checkin_data(train_pos_fname)
    # neg_train_dict, neg_train_businessIds_dict, _ = load_checkin_data(train_neg_fname)

    pos_train_dict, pos_train_businessIds_dict, user_num = load_checkin_data_v1(train_pos_fname, vali_pos_fname)
    neg_train_dict, neg_train_businessIds_dict, _ = load_checkin_data_v1(train_neg_fname, vali_neg_fname)


    n_reference = args.n_reference
    alg = MetaLearningWrapper(learning_rate=args.learning_rate, keep_rate=1.0, margin=args.margin,
                              obv=args.obv, num_query_per_cls=21, # num_query_per_cls depends on the business id selected
                              num_ref_per_cls=n_reference, user_feature_dim=args.user_feature_dim,
                              context_vector_size = args.context_vector_size, num_heads=args.num_heads,
                              loc_feature_dim=args.loc_feature_dim, feature_dim=args.feature_dim,
                              loc_ADJ=location_adjacent_matrix, loc_num=loc_num, user_num=user_num)
    alg.create(training=False)

    chosen_business_id = 270
    pos_vec = [None]
    neg_vec = [None]
    pos_vec[0] = pos_train_dict[chosen_business_id]
    neg_vec[0] = neg_train_dict[chosen_business_id]
    n_pos_examples_train = len(pos_train_dict[chosen_business_id])
    reference_ids = np.random.permutation(n_pos_examples_train)

    reference_vec = [[pos_train_dict[chosen_business_id][_] for _ in reference_ids[:n_reference]]]

    ref_vec_, pos_vec_, neg_vec_ = alg.evaluate_visual(reference_vec, pos_vec, neg_vec)
    out_str = StringIO.StringIO()

    def num2str(vec):
        return [str(item) for item in vec]

    for item in ref_vec_:
        for vec in item:
            out_str.write('0' + ',')
            vec = num2str(vec)
            out_str.write(','.join(vec))
            out_str.write('\n')

    for item in pos_vec_:
        for vec in item:
            out_str.write('1' + ',')
            vec = num2str(vec)
            out_str.write(','.join(vec))
            out_str.write('\n')

    for item in neg_vec_:
        for vec in item:
            out_str.write('2' + ',')
            vec = num2str(vec)
            out_str.write(','.join(vec))
            out_str.write('\n')

    fout = open('embeddings/Tor_' + str(chosen_business_id) + '.txt', 'w')
    fout.write(out_str.getvalue())
    fout.close()


def main():
    if args.mode == 'train':
        train()
    elif args.mode == 'infer':
        infer()
    elif args.mode == 'visual':
        visualize()


if __name__ == '__main__':
    main()
