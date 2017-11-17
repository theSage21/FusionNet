from tqdm import tqdm
import tensorflow as tf


def fullattention(first, b, attention_dim, U, D, scope=''):
    """
    Fully aware attention based on the multiplicative attention
    method. D must be a diagonal matrix.
    Given a batch of time sliced objects, computes attention for them
    """
    # a and b are [BS, DIM]
    second = tf.nn.relu(tf.matmul(U, tf.transpose(b)))
    second = tf.matmul(D, second)  # second half
    # Now we do a dot product
    s = tf.reduce_sum(tf.transpose(tf.multiply(first, second)), axis=1)
    return s


def fuse(howA, howB, attention_dim, scope='', concat=False, B=None):
    """
    Fuses B into A via multiplicative attention.
    Returns the attended representation so that the user
    may then manipulate as they see fit.

    It calculates weights based on howA and howB
    then creates a weighted sum of B based on those weights
    """
    B = B if B is not None else howB
    dim = howA.get_shape().as_list()[-1]
    scope += '_fusion'
    # B = [BS, Time, Dim]
    # A = [BS, Time, Dim]
    with tf.variable_scope(scope):
        a_s = tf.unstack(howA, axis=1)
        b_s = tf.unstack(howB, axis=1)
        fused_A = []
        # -----------------------------------------------------fusion variables
        U = tf.get_variable('U', shape=(attention_dim, dim),
                            dtype=tf.float32)
        I = tf.eye(attention_dim)
        D = tf.get_variable('D', shape=(attention_dim, attention_dim),
                            dtype=tf.float32)
        D = tf.multiply(D, I)  # restrict to diagonal
        for i, a in enumerate(tqdm(a_s, desc='Fuse '+scope)):
            attention_weights = []
            # first half of equation
            first = tf.nn.relu(tf.matmul(U, tf.transpose(a)))
            for j, b in enumerate(b_s):
                s = fullattention(first, b_s[j], attention_dim, U, D, scope)
                attention_weights.append(s)
            attention_weights = tf.stack(attention_weights, axis=1)
            attention_weights = tf.nn.softmax(attention_weights)
            attention_weights = tf.expand_dims(attention_weights, axis=2)
            rep = tf.reduce_sum(tf.multiply(B, attention_weights), axis=1)
            fused_A.append(rep)
        fused_A = tf.stack(fused_A, axis=1)
    return fused_A


def word_fusion(para_glove, q_glove):
    "Fuses glove representations of question into para"
    dim = q_glove.get_shape().as_list()[-1]
    with tf.variable_scope("word_fusion"):
        W = tf.get_variable("W", shape=(dim, dim), dtype=tf.float32)
        p, q = tf.unstack(para_glove, axis=1), tf.unstack(q_glove, axis=1)
        fused_para = []
        for i, pw in enumerate(tqdm(p, desc='word fusion')):
            attention_weights = []
            first = tf.nn.relu(tf.matmul(W, tf.transpose(pw)))
            for j, qw in enumerate(q):
                second = tf.nn.relu(tf.matmul(W, tf.transpose(qw)))
                s = tf.reduce_sum(tf.transpose(tf.multiply(first, second)),
                                  axis=1)
                attention_weights.append(s)
            attention_weights = tf.stack(attention_weights, axis=1)
            attention_weights = tf.nn.softmax(attention_weights)
            attention_weights = tf.expand_dims(attention_weights, axis=2)
            rep = tf.reduce_sum(tf.multiply(q_glove, attention_weights),
                                axis=1)
            fused_para.append(rep)
        fused_para = tf.stack(fused_para, axis=1)
    return fused_para
