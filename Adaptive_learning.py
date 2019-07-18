import numpy as np
import tensorflow as tf
from multi_target_model import *
from dataset import *

flags = tf.flags
FLAGS = flags.FLAGS


class AdaptiveTrainer(object):
    def __init__(self, flags):
        self.FLAGS = flags
        self.model_load_from = "output/model/{0}_to_{1}_pre.pkl".format(flags.source_domain, flags.target_domain)
        self.model_save_to = "output/model/{0}_to_{1}.pkl".format(flags.source_domain, flags.target_domain)

    def train(self, batch, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(self.FLAGS)

        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.total_theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.model_load_from)
            while True:
                R_loss = 0.
                D_loss = 0.
                C_loss = 0.
                P_loss = 0.
                S_loss = 0.
                train_accuracy = 0.
                for b in batch.generate(shuffle=True):
                    x, y, d = zip(*b)
                    _, r_loss = self.sess.run([model.R_solver, model.R_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, d_loss = self.sess.run([model.D_solver, model.D_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, c_loss = self.sess.run([model.C_s_solver, model.C_s_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, p_loss = self.sess.run([model.P_solver, model.P_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, s_loss, accuracy = self.sess.run(
                        [model.S_s_solver, model.S_s_loss, model.accuracy],
                        feed_dict={model.X: x, model.Y: y, model.D: d})
                    R_loss += r_loss
                    D_loss += d_loss
                    C_loss += c_loss
                    P_loss += p_loss
                    S_loss += s_loss
                    train_accuracy += accuracy
                for b in batch.generate(domain='target', shuffle=True):
                    x, y, d = zip(*b)
                    _, r_loss = self.sess.run([model.R_solver, model.R_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, d_loss = self.sess.run([model.D_solver, model.D_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, c_loss = self.sess.run([model.C_t_solver, model.C_t_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, p_loss = self.sess.run([model.P_solver, model.P_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, s_loss, = self.sess.run([model.S_t_solver, model.S_t_loss],
                                               feed_dict={model.X: x, model.Y: y, model.D: d})
                batch_nums = len(batch.x_s) / batch.batch_size
                print('r_loss: {0}, d_loss: {1}, c_loss: {2}, p_loss: {3}, s_loss: {4}, acc: {5}'.format(
                    R_loss / batch_nums,
                    D_loss / batch_nums,
                    C_loss / batch_nums,
                    P_loss / batch_nums,
                    S_loss / batch_nums,
                    train_accuracy / batch_nums
                ))
                # print('train_loss: {0}, train_accuracy: {1}'.format(train_loss / batch_nums, train_accuracy / batch_nums))
                if train_accuracy / batch_nums > 0.:
                    valid_accuracy = model.accuracy.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
                    # pred = model.pred.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
                    # encoding = model.encoding.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(sess=self.sess, save_path=self.model_save_to)
                    else:
                        wait_times += 1
                    if wait_times > self.FLAGS.tolerate_time:
                        print('best_result: {0}'.format(best_result))
                        break
                    # print('pred: {0}'.format(pred))
                    # print('encoding: {0}'.format(encoding))
                    print('valid_accuracy: {0}'.format(valid_accuracy))
            saver.restore(self.sess, self.model_save_to)
            test_accuracy = model.accuracy.eval({model.X: x_test, model.Y: y_test}, session=self.sess)
            print('test_accuracy: {0}'.format(test_accuracy))
            return test_accuracy


def main(_):
    x, y, offset = load_amazon(5000, FLAGS.data_load_from)
    x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst = split_data(
        FLAGS.source_domain, FLAGS.target_domain, x, y, offset, 2000)  # 训练集2000样本

    x = turn_tfidf(np.concatenate([x_s_tr, x_s_tst, x_t_tr, x_t_tst], axis=0))
    x_s = x[:len(x_s_tr) + len(x_s_tst)]
    x_t = x[len(x_s):]

    x_s_tr = np.copy(x_s[:len(x_s_tr)])
    x_s_tst = np.copy(x_s[len(x_s_tr):])  # 源域的测试集用不到

    x_t_tr = np.copy(x_t[:len(x_t_tr)])  # train保持2000不变，test再切分为少样本、验证集和测试集
    x_t_tst = np.copy(x_t[len(x_t_tr):])

    x_t_tune = x_t_tst[:50]
    y_t_tune = y_t_tst[:50]
    x_t_tst = x_t_tst[50:]
    y_t_tst = y_t_tst[50:]

    x_t_valid = x_t_tst[:500]
    y_t_valid = y_t_tst[:500]
    x_t_tst = x_t_tst[500:]
    y_t_tst = y_t_tst[500:]

    batch = Batch(x_s_tr, y_s_tr, x_t_tr, FLAGS.batch_size)
    # for b in batch.generate(shuffle=False):
    #     x, y, d = zip(*b)
    #     print(y)
    #     break
    trainer = AdaptiveTrainer(FLAGS)
    trainer.train(batch, x_t_valid, y_t_valid, x_t_tst, y_t_tst)


flags.DEFINE_string("data_load_from", "data/amazon.mat", "data path")
flags.DEFINE_integer("source_domain", 3, "source domain id")
flags.DEFINE_integer("target_domain", 0, "target domain id")
flags.DEFINE_integer("n_domains", 2, "number of domains")
flags.DEFINE_integer("tolerate_time", 20, "stop training if it exceeds tolerate time")
flags.DEFINE_integer("n_input", 5000, "size of input data")
flags.DEFINE_integer("n_classes", 2, "size of output data")
flags.DEFINE_integer("n_hidden_s", 50, "size of shared encoder hidden layer")
flags.DEFINE_integer("n_hidden_p", 50, "size of private encoder hidden layer")
flags.DEFINE_integer("batch_size", 50, "batch size")
flags.DEFINE_float("lr", 1e-4, "learning rate")

if __name__ == "__main__":
    tf.app.run()
