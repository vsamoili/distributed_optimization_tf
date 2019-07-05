import tensorflow as tf

class Worker:
        def __init__(self, _id):
                self._id = _id
        def get_vars(self, pv):
                self.vars_ = pv
        def get_grads(self, loss, xs):
                self.grads = tf.gradients(loss, xs)
