import numpy as np
import tensorflow as tf


class Batch(object):
    def __init__(self, source_x, source_y, target_x, batch_size):
        self.x_s = source_x
        self.y_s = source_y
        self.x_t = target_x
        self.batch_size = batch_size

    def generate(self, domain='source', shuffle=False):
        d_s = np.tile(np.array([0., 1]), (len(self.x_s), 1))  # 源域的域标签
        d_t = np.tile(np.array([1., 0]), (len(self.x_t), 1))  # 目标域的域标签
        y_t = np.tile(np.array([0., 0]), (len(self.x_t), 1))  # 目标域的分类任务标签，有问题，损失为0梯度不一定为0
        # x = np.vstack((self.x_s, self.x_t))
        # y = np.vstack((self.y_s, y_t))
        # d = np.vstack((d_s, d_t))
        # x = self.x_s
        # y = self.y_s
        # d = d_s
        if domain == 'source':
            x = self.x_s
            y = self.y_s
            d = d_s
        else:
            x = self.x_t
            y = y_t
            d = d_t

        if shuffle:
            shuffled_data = np.random.permutation(list(zip(x, y, d)))
        else:
            shuffled_data = list(zip(x, y, d))

        sample_nums = len(x)
        batch_nums = int(sample_nums / self.batch_size)
        for start in range(batch_nums):
            yield shuffled_data[start * self.batch_size: min((start + 1) * self.batch_size, sample_nums)]


class AdaptiveModel(object):
    def __init__(self, flags):
        self.FLAGS = flags
        self.gamma_r = 1.0
        self.gamma_c = 0.5
        self.gamma_d = 0.1

    def build_model(self):
        # tf graph input
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_input])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_classes])
        self.D = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_domains])

        # fd network
        self.shared_encode_mlp = MLP('shared_encode_mlp', [self.FLAGS.n_input, self.FLAGS.n_hidden_s],
                                     [tf.nn.sigmoid])
        self.private_encode_mlp = MLP('private_encode_mlp', [self.FLAGS.n_input, self.FLAGS.n_hidden_p],
                                      [tf.nn.sigmoid])
        self.shared_decode_mlp = MLP('shared_decode_mlp',
                                     [self.FLAGS.n_hidden_s, (self.FLAGS.n_hidden_s + self.FLAGS.n_input) // 2,
                                      self.FLAGS.n_input],
                                     [tf.nn.tanh, tf.nn.relu])
        self.private_decode_mlp = MLP('private_decode_mlp',
                                      [self.FLAGS.n_hidden_p, (self.FLAGS.n_hidden_p + self.FLAGS.n_input) // 2,
                                       self.FLAGS.n_input],
                                      [tf.nn.tanh, tf.nn.relu])
        self.domain_output_mlp = MLP('domain_output_mlp', [self.FLAGS.n_hidden_s, self.FLAGS.n_domains],
                                     [identity])
        self.task_output_mlp = MLP('task_output_mlp', [self.FLAGS.n_hidden_s, self.FLAGS.n_classes],
                                   [identity])

        encoding_s = self.shared_encode_mlp.apply(self.X)  # 编码共享表示
        encoding_p = self.private_encode_mlp.apply(self.X)  # 编码私有表示
        self.total_theta = (self.shared_encode_mlp.parameters +
                            self.private_encode_mlp.parameters +
                            self.shared_decode_mlp.parameters +
                            self.private_decode_mlp.parameters +
                            self.domain_output_mlp.parameters +
                            self.task_output_mlp.parameters)
        l2_norm = get_l2_norm(self.total_theta)

        # optimizing the parameters of the decoder F
        decoding_s = self.shared_decode_mlp.apply(encoding_s)  # 共享表示解码器
        decoding_p = self.private_decode_mlp.apply(encoding_p)  # 私有表示解码器
        decoding = decoding_s + decoding_p
        self.theta_f = self.shared_decode_mlp.parameters + self.private_decode_mlp.parameters
        self.R_loss = self.gamma_r * tf.reduce_mean(tf.square(decoding - self.X)) + l2_norm  # 重建损失只用来优化解码器
        self.R_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.R_loss,
                                                                                     var_list=self.theta_f)

        # optimizing the parameters of the domain classifier D
        pred_s = self.domain_output_mlp.apply(encoding_s)  # 共享表示领域判别
        pred_p = self.domain_output_mlp.apply(encoding_p)  # 私有表示领域判别
        loss_s = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.D, logits=pred_s))
        loss_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.D, logits=pred_p))
        self.D_loss = self.gamma_d * (loss_s + loss_p) + l2_norm
        self.theta_d = self.domain_output_mlp.parameters
        self.D_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.D_loss,
                                                                                     var_list=self.theta_d)

        # optimizing the parameters of the label classifier C
        pred = self.task_output_mlp.apply(encoding_s)
        self.pred = pred
        self.prob = tf.nn.softmax(pred)
        mean_pred = tf.tile(
            tf.reshape(tf.reduce_mean(pred, axis=0), [-1, 2]), [self.FLAGS.batch_size, 1])
        # source_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred)) \
        #               / (tf.reduce_sum(self.Y) + tf.constant(1e-8, dtype=tf.float32))
        # mask = 1. - tf.tile(tf.reshape(tf.reduce_sum(self.Y, axis=-1),
        #                                [-1, 1]),
        #                     [1, self.FLAGS.n_classes])  # 源域样例不计算信息熵，只计算交叉熵
        # target_loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(mask * pred), logits=pred))  # 用v2合适吗？
        self.C_s_loss = self.gamma_c * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred)) + l2_norm
        self.C_t_loss = self.gamma_c * (
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(pred), logits=pred)) -
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(pred), logits=mean_pred))) + l2_norm
        # self.C_t_loss = self.gamma_c * tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(pred), logits=pred))
        # self.C_loss = self.gamma_c * (source_loss) + l2_norm  # 目标域任务标签为全0向量，对应交叉熵损失为0
        self.theta_c = self.task_output_mlp.parameters
        self.C_s_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.C_s_loss,
                                                                                       var_list=self.theta_c)
        self.C_t_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.C_t_loss,
                                                                                       var_list=self.theta_c)
        correct_prediction = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(pred, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # optimizing the parameters of the private encoder Ep
        self.P_loss = self.R_loss #+ self.gamma_d * loss_p
        self.theta_p = self.private_encode_mlp.parameters
        self.P_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.P_loss,
                                                                                     var_list=self.theta_p)

        # optimizing the parameters of the shared encoder Es
        self.theta_s = self.shared_encode_mlp.parameters
        # self.S_loss = self.R_loss + self.C_s_loss - self.gamma_d * loss_s  # 可能出错的地方
        self.S_s_loss = self.C_s_loss + self.R_loss - self.gamma_d * loss_s - l2_norm
        self.S_t_loss =  self.R_loss - self.gamma_d * loss_s - l2_norm + self.C_t_loss
        self.S_s_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.S_s_loss,
                                                                                       var_list=self.theta_s)
        self.S_t_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.S_t_loss,
                                                                                       var_list=self.theta_s)
        # self.S_solver = tf.train.RMSPropOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.S_loss,
        #                                                                                 var_list=self.theta_s)


class MLP(object):
    def __init__(self, name, dims, activations):
        self.name = name
        self.dims = dims
        self.activations = activations
        self.weights = []
        self.biases = []
        self._initialize()

    @property
    def parameters(self):
        return self.weights + self.biases

    def _initialize(self):
        for i in range(len(self.dims) - 1):
            w = tf.Variable(xavier_init([self.dims[i], self.dims[i + 1]]), name=self.name + '_w_{0}'.format(i))
            b = tf.Variable(xavier_init([self.dims[i + 1]]), name=self.name + '_b_{0}'.format(i))
            self.weights.append(w)
            self.biases.append(b)

    def apply(self, x):
        out = x
        for a, w, b in zip(self.activations, self.weights, self.biases):
            out = a(tf.add(tf.matmul(out, w), b))
        return out


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)  # xavier初始化方法的标准差计算公式，置out_dim=0
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def identity(x):
    return x


def get_l2_norm(theta):
    l2_norm = 0.
    for tensor in theta:
        l2_norm += tf.reduce_sum(tf.abs(tensor))
    return 0.0001 * l2_norm
