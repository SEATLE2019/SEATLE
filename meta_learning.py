import math
import numpy as np
import random
import time
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import batch_norm


class MetaLearning:
    def __init__(self, **kwargs):
        self.lr = kwargs['learning_rate']
        self.kr = kwargs['keep_rate']
        self.margin_v = kwargs['margin']
        self.user_feature_dim = kwargs['user_feature_dim']
        self.loc_feature_dim = kwargs['loc_feature_dim']
        self.feature_dim = kwargs['feature_dim']
        self.context_vector_size = kwargs['context_vector_size']
        self.loc_ADJ = kwargs['loc_ADJ']
        self.loc_ADJ = tf.cast(self.loc_ADJ, tf.float32)
        self.loc_num = kwargs['loc_num']
        self.user_num = kwargs['user_num']
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.loc_initial_feature_size = 10
        self.loc_gcn_output_feature_num = 2
        self.neuron_per_layer = [5]

        # The following parameters are for multi-head attention.
        # We assume d_v always equals d_k
        self.d_model = kwargs['feature_dim']
        self.h = kwargs['num_heads']
        assert self.d_model % self.h == 0
        self.d_k = self.d_model // self.h
        self.attn = None

        self.training = False
        self.init_embedding()
        self.saver = tf.train.Saver(tf.global_variables())


    def similarityCalculation(self, referece_inputs, query_inputs):
        with tf.variable_scope('similarity', reuse=tf.AUTO_REUSE):
            # use the dot product as the similarity metric
            scores = tf.squeeze(tf.matmul(referece_inputs, tf.transpose(query_inputs, [0, 2, 1])), [1])
            # normalize the scores
            scores = tf.nn.sigmoid(scores)   # sigmoid in [0, 1]
            #scores = tf.nn.tanh(scores)  # tanh in [-1, 1]
            return scores


    def computeCenterWithAttention(self, r_vec):
        with tf.variable_scope('reference_center', reuse=tf.AUTO_REUSE):
            # [classes x instances x dim]
            inputs = r_vec
            # [classes x instances x dim]
            nn_outputs = tf.tanh(tf.einsum("aij,jk->aik", inputs, self.attention_layer_W))
            # [batch_size x instances]
            weights = tf.nn.softmax(tf.einsum("aij,j->ai", nn_outputs, self.reference_context))
            # [classes x dim]
            result = tf.einsum("aij,ai->aj", inputs, weights)
            return result


    def representationLearning(self, feature_vec, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            input = feature_vec
            hidden_outputs = tf.nn.xw_plus_b(input, self.W, self.b, name="hidden_outputs")
            outputs = tf.tanh(hidden_outputs)
            return outputs


    def similarity(self, feature_vec, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            input = feature_vec
            hidden_outputs = tf.nn.xw_plus_b(input, self.W_similarity, self.b_similarity, name="similarity_layer")
            outputs = tf.tanh(hidden_outputs)
            return outputs


    def attention(self, query, key, value):
        # "Args:query: [batch_size, n, d_k], key.shape = [batch_size, m, d_k], value.shape = [batch_size, m, d_v]"
        # "Output:output.shape = [batch_size, n, d_k]"
        "Compute 'Scaled Dot Product Attention'"
        batch_size, n_query, d_k = query.get_shape().as_list()
        scores = tf.matmul(query, tf.transpose(key, [0, 2, 1])) / math.sqrt(d_k)  # n*m
        p_attn = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(p_attn, value), p_attn


    def multiHeadedAttention(self, query, key, value, reuse, mask=None):
        with tf.variable_scope('interaction_learning', reuse=reuse):
            # 1) Do all the linear projections in batch from d_model => h x d_k
            batch_size, n_query, _ = tf.unstack(tf.shape(query))
            batch_size, n_key, _ = tf.unstack(tf.shape(key))
            batch_size, n_value, _ = tf.unstack(tf.shape(value))
            proj_query = tf.reshape(tf.layers.dense(query, self.d_model,
                                                    kernel_regularizer=self.regularizer, name='output_query'),
                                    [-1, n_query, self.h, self.d_k])
            proj_query = tf.reshape(tf.transpose(proj_query, [0, 2, 1, 3]), [-1, n_query, self.d_k])

            proj_key = tf.reshape(tf.layers.dense(key, self.d_model,
                                                  kernel_regularizer=self.regularizer, name='output_key'),
                                  [-1, n_key, self.h, self.d_k])
            proj_key = tf.reshape(tf.transpose(proj_key, [0, 2, 1, 3]), [-1, n_key, self.d_k])

            proj_value = tf.reshape(tf.layers.dense(value, self.d_model,
                                                    kernel_regularizer=self.regularizer, name='output_value'),
                                    [-1, n_value, self.h, self.d_k])
            proj_value = tf.reshape(tf.transpose(proj_value, [0, 2, 1, 3]), [-1, n_value, self.d_k])

            # 2) Apply attention on all the projected vectors in batch.
            x, self.attn = self.attention(proj_query, proj_key, proj_value)  # [batch_size*n_head, n_query, d_v]

            # 3) "Concat" using a view and apply a final linear.
            x = tf.reshape(x, [-1, self.h, n_query, self.d_k])
            x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, n_query, n_head, d_v]
            x = tf.reshape(x, [-1, n_query, self.h * self.d_k])
            return tf.layers.dense(x, self.d_model, activation=None, trainable=True,
                                   kernel_regularizer=self.regularizer, name='mult_head', reuse=reuse)


    def PositionwiseFeedForward(self, x, d_model, d_diff, reuse):
        l1 = tf.layers.dense(x, d_diff, activation=tf.nn.relu, kernel_regularizer=self.regularizer, name='feed1',
                             reuse=reuse)
        # l1 = tf.dropout(l1, keep_rate = 0.8)
        l2 = tf.layers.dense(l1, d_model, kernel_regularizer=self.regularizer, name='feed2', reuse=reuse)
        return l2


    def SublayerConnection(self, x, keep_rate):
        return batch_norm((x + tf.dropout(self.model(x, x, x), keep_rate=keep_rate)))


    def init_embedding(self, scope='initial_scope'):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.business_vocab_size = self.loc_num + 10
                self.user_vocab_size = self.user_num + 10
                self.business_embedding_matrix = tf.get_variable(
                    name="business_embedding_matrix",
                    shape=[self.business_vocab_size, self.loc_feature_dim],
                    initializer=layers.xavier_initializer(),
                    dtype=tf.float32)
                self.user_embedding_matrix = tf.get_variable(
                    name="user_embedding_matrix",
                    shape=[self.user_vocab_size, self.user_feature_dim],
                    initializer=layers.xavier_initializer(),
                    dtype=tf.float32)

                self.attention_layer_W = tf.get_variable(name="attention_nn_layer_weights",
                                                         shape=[self.feature_dim, self.context_vector_size],
                                                         initializer=layers.xavier_initializer())
                self.reference_context = tf.get_variable(name="reference_context",
                                                         shape=[self.context_vector_size],
                                                         initializer=layers.xavier_initializer())

                self.W = tf.get_variable(
                    name="user_business_interaction_W",
                    shape=[self.loc_feature_dim + self.user_feature_dim + 4, self.feature_dim], # 4 geo features
                    initializer=layers.xavier_initializer(),
                    dtype=tf.float32)
                self.b = tf.get_variable(
                    name="user_business_interaction_b",
                    shape=[self.feature_dim],
                    initializer=tf.zeros_initializer())


    def model(self, num_query_per_cls=3, num_ref_per_cls=3, output_dim=10):
        with tf.variable_scope('my_model') as scope:
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            # Positive query part
            self.p_businessId = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.int32, name='p_businessIds')
            self.p_userId = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.int32, name='p_userIds')
            self.p_geo_f0 = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.float32, name='p_geo_0')
            self.p_geo_f1 = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.float32, name='p_geo_near_1')
            self.p_geo_f2 = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.float32, name='p_geo_far_2')
            self.p_geo_f3 = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.float32, name='p_geo_mean_3')

            p_business_embedded = tf.nn.embedding_lookup(self.business_embedding_matrix, self.p_businessId)
            p_user_embedded = tf.nn.embedding_lookup(self.user_embedding_matrix, self.p_userId)
            self.p_geo_f0_ = tf.expand_dims(self.p_geo_f0, axis=-1)
            self.p_geo_f1_ = tf.expand_dims(self.p_geo_f1, axis=-1)
            self.p_geo_f2_ = tf.expand_dims(self.p_geo_f2, axis=-1)
            self.p_geo_f3_ = tf.expand_dims(self.p_geo_f3, axis=-1)
            p_input_vec = tf.concat(
                [p_business_embedded, p_user_embedded, self.p_geo_f0_, self.p_geo_f1_, self.p_geo_f2_, self.p_geo_f3_],
                axis=-1)

            # Negative query part
            self.n_businessId = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.int32, name='n_businessIds')
            self.n_userId = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.int32, name='n_userIds')
            self.n_geo_f0 = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.float32, name='n_geo_0')
            self.n_geo_f1 = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.float32, name='n_geo_near_1')
            self.n_geo_f2 = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.float32, name='n_geo_far_2')
            self.n_geo_f3 = tf.placeholder(shape=(None, num_query_per_cls), dtype=tf.float32, name='n_geo_mean_3')

            n_business_embedded = tf.nn.embedding_lookup(self.business_embedding_matrix, self.n_businessId)
            n_user_embedded = tf.nn.embedding_lookup(self.user_embedding_matrix, self.n_userId)
            self.n_geo_f0_ = tf.expand_dims(self.n_geo_f0, axis=-1)
            self.n_geo_f1_ = tf.expand_dims(self.n_geo_f1, axis=-1)
            self.n_geo_f2_ = tf.expand_dims(self.n_geo_f2, axis=-1)
            self.n_geo_f3_ = tf.expand_dims(self.n_geo_f3, axis=-1)
            n_input_vec = tf.concat(
                [n_business_embedded, n_user_embedded, self.n_geo_f0_, self.n_geo_f1_, self.n_geo_f2_, self.n_geo_f3_],
                axis=-1)

            # Reference part
            self.r_businessId = tf.placeholder(shape=(None, num_ref_per_cls), dtype=tf.int32, name='q_businessIds')
            self.r_userId = tf.placeholder(shape=(None, num_ref_per_cls), dtype=tf.int32, name='q_userIds')
            self.r_geo_f0 = tf.placeholder(shape=(None, num_ref_per_cls), dtype=tf.float32, name='q_geo_0')
            self.r_geo_f1 = tf.placeholder(shape=(None, num_ref_per_cls), dtype=tf.float32, name='q_geo_near_1')
            self.r_geo_f2 = tf.placeholder(shape=(None, num_ref_per_cls), dtype=tf.float32, name='q_geo_far_2')
            self.r_geo_f3 = tf.placeholder(shape=(None, num_ref_per_cls), dtype=tf.float32, name='q_geo_mean_3')

            r_business_embedded = tf.nn.embedding_lookup(self.business_embedding_matrix, self.r_businessId)
            r_user_embedded = tf.nn.embedding_lookup(self.user_embedding_matrix, self.r_userId)
            self.r_geo_f0_ = tf.expand_dims(self.r_geo_f0, axis=-1)
            self.r_geo_f1_ = tf.expand_dims(self.r_geo_f1, axis=-1)
            self.r_geo_f2_ = tf.expand_dims(self.r_geo_f2, axis=-1)
            self.r_geo_f3_ = tf.expand_dims(self.r_geo_f3, axis=-1)
            r_input_vec = tf.concat(
                [r_business_embedded, r_user_embedded, self.r_geo_f0_, self.r_geo_f1_, self.r_geo_f2_, self.r_geo_f3_],
                axis=-1)


            # Construct the initial embeddings for postive queries
            dim_b, dim_i, dim_f = p_input_vec.get_shape().as_list()
            p_input_vec = tf.reshape(p_input_vec, [-1, dim_f])
            p_input_vec = self.representationLearning(p_input_vec, scope)
            p_input_vec = tf.reshape(p_input_vec, [-1, dim_i, output_dim])

            # Construct the initial embeddings for ngeative queries
            dim_b, dim_i, dim_f = n_input_vec.get_shape().as_list()
            n_input_vec = tf.reshape(n_input_vec, [-1, dim_f])
            n_input_vec = self.representationLearning(n_input_vec, scope)
            n_input_vec = tf.reshape(n_input_vec, [-1, dim_i, output_dim])

            # Construct the initial mebddings for references
            dim_b, dim_i, dim_f = r_input_vec.get_shape().as_list()
            r_input_vec = tf.reshape(r_input_vec, [-1, dim_f])
            r_input_vec = self.representationLearning(r_input_vec, scope)
            r_input_vec = tf.reshape(r_input_vec, [-1, dim_i, output_dim])

            # Apply self-attention
            self.p_r_multihead_ = self.multiHeadedAttention(query=p_input_vec, key=r_input_vec, value=r_input_vec,
                                                            reuse=False)
            self.n_r_multihead_ = self.multiHeadedAttention(query=n_input_vec, key=r_input_vec, value=r_input_vec,
                                                            reuse=True)

            # Residual shortcut connection
            p_combined_embedding = p_input_vec + self.p_r_multihead_
            n_combined_embedding = n_input_vec + self.n_r_multihead_

            # Calculate the representative of references
            r_center_vec = self.computeCenterWithAttention(r_input_vec)
            r_center_vec = tf.expand_dims(r_center_vec, axis=1)

            # Calculate the similarity Scores
            self.p_score_ = self.similarityCalculation(r_center_vec, p_combined_embedding)
            self.n_score_ = self.similarityCalculation(r_center_vec, n_combined_embedding)

            self.p_score = tf.reshape(self.p_score_, [-1])
            self.n_score = tf.reshape(self.n_score_, [-1])


            # For the purpose of visualization
            self.ref_vec_ = r_input_vec
            self.pos_vec_ = p_input_vec
            self.neg_vec_ = n_input_vec

            # Apply hinge loss as the ranking loss
            self.loss_v = tf.maximum(0.0, self.n_score - self.p_score + self.margin_v)
            self.loss = tf.reduce_mean(self.loss_v)

            # Compute the accuracy
            correct_pred_representation = tf.ones_like(self.p_score)
            incorrect_pred_representation = tf.zeros_like(self.p_score)
            preds = tf.where(
                tf.greater(self.p_score, self.n_score),
                x=correct_pred_representation,
                y=incorrect_pred_representation,
            )
            correct_preds = tf.equal(preds, correct_pred_representation)
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"), name="accuracy")

            self.train_op = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(self.lr, self.global_step,
                                                                                            decay_steps=1000,
                                                                                            decay_rate=0.95,
                                                                                            staircase=True)).minimize(
                self.loss, global_step=self.global_step)

            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            init = [init_g, init_l]
        return self.p_businessId, self.p_userId, self.p_geo_f0, self.p_geo_f1, self.p_geo_f2, self.p_geo_f3, \
               self.n_businessId, self.n_userId, self.n_geo_f0, self.n_geo_f1, self.n_geo_f2, self.n_geo_f3, \
               self.r_businessId, self.r_userId, self.r_geo_f0, self.r_geo_f1, self.r_geo_f2, self.r_geo_f3, \
               self.loss, self.accuracy, self.global_step, self.train_op, init, self.saver, self.p_score, self.n_score,\
               self.ref_vec_, self.pos_vec_, self.neg_vec_


class MetaLearningWrapper:
    def __init__(self, learning_rate=1e-3, keep_rate=0.9, margin=1e-7, obv=10, num_query_per_cls=3,
                 num_ref_per_cls=3, user_feature_dim=10, loc_feature_dim=10, feature_dim=10, num_heads=2,
                 context_vector_size = 10, loc_ADJ=None, loc_num=1, user_num=1):
        self.obv = obv
        self.lr = learning_rate
        self.kr = keep_rate
        self.margin_ = margin
        self.training = True

        self.total_loss = 0
        self.total_acc = 0.0

        self.metaL = MetaLearning(learning_rate=self.lr, keep_rate=self.kr, margin=self.margin_,
                                  user_feature_dim=user_feature_dim, loc_feature_dim=loc_feature_dim,
                                  feature_dim=feature_dim, context_vector_size= context_vector_size,
                                  num_heads=num_heads, loc_ADJ=loc_ADJ, loc_num=loc_num, user_num=user_num)

        self.p_businessId, self.p_userId, self.p_geo_f0, self.p_geo_f1, self.p_geo_f2, self.p_geo_f3, \
        self.n_businessId, self.n_userId, self.n_geo_f0, self.n_geo_f1, self.n_geo_f2, self.n_geo_f3, \
        self.r_businessId, self.r_userId, self.r_geo_f0, self.r_geo_f1, self.r_geo_f2, self.r_geo_f3, \
        self.loss, self.acc, self.global_step, self.train_op, self.init_, self.saver, self.p_s, self.n_s,\
        self.ref_vec_, self.pos_vec_, self.neg_vec_\
            = self.metaL.model(num_query_per_cls=num_query_per_cls, num_ref_per_cls=num_ref_per_cls,
                               output_dim=feature_dim)


    def create(self, training=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.log_device_placement = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        # tf.set_random_seed(123)
        self.metaL.training = training
        if training:
            self.sess.run(self.init_)
        else:
            self.sess.run(self.init_)
            checkpoint = tf.train.get_checkpoint_state('Model/')
            print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

        print('create Meta Learning algorithm')


    def saveModel(self, folderName, step):
        self.saver.save(self.sess, folderName, step)


    def constructVector(self, input):
        num_business = len(input)
        businessId = [[] for _ in range(num_business)]
        userId, geo_f0, geo_f1, geo_f2, geo_f3 = [[] for _ in range(num_business)], [[] for _ in range(num_business)], [
            [] for _ in range(num_business)], [[] for _ in range(num_business)], [[] for _ in range(num_business)]
        for idx, each_business in enumerate(input):
            for each_instance in each_business:
                businessId[idx].append(each_instance[0])
                userId[idx].append(each_instance[1])
                geo_f0[idx].append(each_instance[2])
                geo_f1[idx].append(each_instance[3])
                geo_f2[idx].append(each_instance[4])
                geo_f3[idx].append(each_instance[5])

        return businessId, userId, geo_f0, geo_f1, geo_f2, geo_f3


    def updateParameters(self, r_vec, pos_vec, neg_vec, eval=False, shuffle=True):
        if shuffle:
            for business_ins in pos_vec:
                random.shuffle(business_ins)

        r_businessId, r_userId, r_geo_f0, r_geo_f1, r_geo_f2, r_geo_f3 = self.constructVector(r_vec)
        p_businessId, p_userId, p_geo_f0, p_geo_f1, p_geo_f2, p_geo_f3 = self.constructVector(pos_vec)
        n_businessId, n_userId, n_geo_f0, n_geo_f1, n_geo_f2, n_geo_f3 = self.constructVector(neg_vec)

        '''
        p_businessId = [1, 2, 3]  # version 1
        p_businessId = [[1,1,1], [2,2,2], [3,3,3]] # current version
        p_userId = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        p_geo_f0 = [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
        p_geo_f1 = [[4.0, 4.1, 4.2], [5.0, 5.1, 5.2], [6.0, 6.1, 6.2]]
        p_geo_f2 = [[7.0, 7.1, 7.2], [8.0, 8.1, 8.2], [9.0, 9.1, 9.2]]
        p_geo_f3 = [[10.0, 10.1, 10.2], [11.0, 11.1, 11.2], [12.0, 12.1, 12.2]]

        n_businessId = [1, 2, 3]
        n_userId = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        n_geo_f0 = [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
        n_geo_f1 = [[4.0, 4.1, 4.2], [5.0, 5.1, 5.2], [6.0, 6.1, 6.2]]
        n_geo_f2 = [[7.0, 7.1, 7.2], [8.0, 8.1, 8.2], [9.0, 9.1, 9.2]]
        n_geo_f3 = [[10.0, 10.1, 10.2], [11.0, 11.1, 11.2], [12.0, 12.1, 12.2]]

        r_businessId = [1, 2, 3]
        r_userId = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        r_geo_f0 = [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
        r_geo_f1 = [[4.0, 4.1, 4.2], [5.0, 5.1, 5.2], [6.0, 6.1, 6.2]]
        r_geo_f2 = [[7.0, 7.1, 7.2], [8.0, 8.1, 8.2], [9.0, 9.1, 9.2]]
        r_geo_f3 = [[10.0, 10.1, 10.2], [11.0, 11.1, 11.2], [12.0, 12.1, 12.2]]
        '''

        t_start = time.clock()
        feed_dict = {self.p_businessId: p_businessId, self.p_userId: p_userId, self.p_geo_f0: p_geo_f0,
                     self.p_geo_f1: p_geo_f1, self.p_geo_f2: p_geo_f2, self.p_geo_f3: p_geo_f3,
                     self.n_businessId: n_businessId, self.n_userId: n_userId, self.n_geo_f0: n_geo_f0,
                     self.n_geo_f1: n_geo_f1, self.n_geo_f2: n_geo_f2, self.n_geo_f3: n_geo_f3,
                     self.r_businessId: r_businessId, self.r_userId: r_userId, self.r_geo_f0: r_geo_f0,
                     self.r_geo_f1: r_geo_f1, self.r_geo_f2: r_geo_f2, self.r_geo_f3: r_geo_f3}

        self.time, loss_, _, acc, p, n = self.sess.run(
            [self.global_step, self.loss, self.train_op, self.acc, self.p_s, self.n_s],
            feed_dict=feed_dict)
        t_span = time.clock() - t_start

        self.total_loss += loss_
        self.total_acc += acc

        if self.time % self.obv == 0 and eval:
            print('Step: ', self.time,
                  'loss: ', np.round(self.total_loss / self.obv, 7),
                  'accuracy: ', np.round(self.total_acc / self.obv, 5),
                  'time: ', round(t_span, 4))
            self.total_loss = 0
            self.total_acc = 0.0


    def evaluate(self, r_vec, pos_vec, neg_vec, shuffle=False):
        if shuffle:
            for business_ins in pos_vec:
                random.shuffle(business_ins)
        r_businessId, r_userId, r_geo_f0, r_geo_f1, r_geo_f2, r_geo_f3 = self.constructVector(r_vec)
        p_businessId, p_userId, p_geo_f0, p_geo_f1, p_geo_f2, p_geo_f3 = self.constructVector(pos_vec)
        n_businessId, n_userId, n_geo_f0, n_geo_f1, n_geo_f2, n_geo_f3 = self.constructVector(neg_vec)

        feed_dict = {self.p_businessId: p_businessId, self.p_userId: p_userId, self.p_geo_f0: p_geo_f0,
                     self.p_geo_f1: p_geo_f1, self.p_geo_f2: p_geo_f2, self.p_geo_f3: p_geo_f3,
                     self.n_businessId: n_businessId, self.n_userId: n_userId, self.n_geo_f0: n_geo_f0,
                     self.n_geo_f1: n_geo_f1, self.n_geo_f2: n_geo_f2, self.n_geo_f3: n_geo_f3,
                     self.r_businessId: r_businessId, self.r_userId: r_userId, self.r_geo_f0: r_geo_f0,
                     self.r_geo_f1: r_geo_f1, self.r_geo_f2: r_geo_f2, self.r_geo_f3: r_geo_f3}

        loss, acc, pos_score, neg_score = self.sess.run([self.loss, self.acc, self.p_s, self.n_s], feed_dict=feed_dict)
        return loss, acc, pos_score, neg_score


    def evaluate_visual(self, r_vec, pos_vec, neg_vec):
        r_businessId, r_userId, r_geo_f0, r_geo_f1, r_geo_f2, r_geo_f3 = self.constructVector(r_vec)
        p_businessId, p_userId, p_geo_f0, p_geo_f1, p_geo_f2, p_geo_f3 = self.constructVector(pos_vec)
        n_businessId, n_userId, n_geo_f0, n_geo_f1, n_geo_f2, n_geo_f3 = self.constructVector(neg_vec)

        feed_dict = {self.p_businessId: p_businessId, self.p_userId: p_userId, self.p_geo_f0: p_geo_f0,
                     self.p_geo_f1: p_geo_f1, self.p_geo_f2: p_geo_f2, self.p_geo_f3: p_geo_f3,
                     self.n_businessId: n_businessId, self.n_userId: n_userId, self.n_geo_f0: n_geo_f0,
                     self.n_geo_f1: n_geo_f1, self.n_geo_f2: n_geo_f2, self.n_geo_f3: n_geo_f3,
                     self.r_businessId: r_businessId, self.r_userId: r_userId, self.r_geo_f0: r_geo_f0,
                     self.r_geo_f1: r_geo_f1, self.r_geo_f2: r_geo_f2, self.r_geo_f3: r_geo_f3}

        ref_vec, pos_vec, neg_vec = self.sess.run([self.ref_vec_, self.pos_vec_, self.neg_vec_], feed_dict=feed_dict)
        return ref_vec, pos_vec, neg_vec
