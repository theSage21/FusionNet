import logging
from tqdm import tqdm
import tensorflow as tf
from parts import fuse, word_fusion, timedrop


logging.getLogger().setLevel("DEBUG")
birnn = tf.nn.bidirectional_dynamic_rnn


def build(*, batchsize, max_p_len, glove_dim,
          cove_dim, max_q_len, sl_att_dim,
          nerpos_dim, tf_dim, reading_rep_dim,
          final_ques_under_dim, sh_att_dim,
          su_att_dim, fully_fused_para_dim,
          selfboost_att_dim, selfboost_rep_dim,
          dropout_proba, is_train, **extras):
    main_scope = 'Training' if is_train else 'Testing'
    with tf.variable_scope(main_scope):
        drop_p = 1.0 if not is_train else dropout_proba
        # ---------------------reading
        logging.info("Defining inputs")
        # First we define shapes for the inputs we need
        p_g_sh = (batchsize, max_p_len, glove_dim)  # para, glove
        q_g_sh = (batchsize, max_q_len, glove_dim)  # ques, glove
        p_c_sh = (batchsize, max_p_len, cove_dim)  # para, cove
        q_c_sh = (batchsize, max_q_len, cove_dim)  # ques, cove
        p_ner_sh = (batchsize, max_p_len, nerpos_dim)  # para, ner + pos
        p_tf_sh = (batchsize, max_p_len, tf_dim)  # para, normalized term freq
        p_em_sh = (batchsize, max_p_len, 1)  # para, exact word match in q

        # we generate the placeholders based on the shapes defined
        inp_para_glove = tf.placeholder(shape=p_g_sh, dtype=tf.float32)
        inp_ques_glove = tf.placeholder(shape=q_g_sh, dtype=tf.float32)
        inp_para_cove = tf.placeholder(shape=p_c_sh, dtype=tf.float32)
        inp_ques_cove = tf.placeholder(shape=q_c_sh, dtype=tf.float32)
        para_nerpos = tf.placeholder(shape=p_ner_sh, dtype=tf.float32)
        para_tf = tf.placeholder(shape=p_tf_sh, dtype=tf.float32)
        para_em = tf.placeholder(shape=p_em_sh, dtype=tf.float32)
        # -------------------embeddings dropout
        para_glove = timedrop(inp_para_glove, drop_p, 'paraGlove')
        para_cove = timedrop(inp_para_cove, drop_p, 'paraCove')
        ques_glove = timedrop(inp_ques_glove, drop_p, 'quesGlove')
        ques_cove = timedrop(inp_ques_cove, drop_p, 'quesCove')

        # TODO: answer placeholder for training
        logging.info("Word level infusion")
        # fused_a = fuse(para_glove, ques_glove, attention_dim, 'test')
        para_q_fused_glove = word_fusion(para_glove, ques_glove)
        para_w_rep = tf.concat([para_glove, para_cove,
                                para_nerpos, para_tf],
                               axis=2)
        ques_w_rep = tf.concat([ques_glove, ques_cove],
                               axis=2)
        para_enhanced_rep = tf.concat([para_w_rep, para_em,
                                       para_q_fused_glove],
                                      axis=2)

        # ---------------------reading
        logging.info("Building Reading section")

        with tf.variable_scope("Reading"):
            f_read_q_low = tf.contrib.rnn.LSTMCell(reading_rep_dim//2)
            b_read_q_low = tf.contrib.rnn.LSTMCell(reading_rep_dim//2)
            ques_low_h, _ = birnn(cell_fw=f_read_q_low, cell_bw=b_read_q_low,
                                  inputs=ques_w_rep, dtype=tf.float32,
                                  scope='ques_low_under')
            ques_low_h = tf.concat(ques_low_h, axis=2)

            f_read_q_high = tf.contrib.rnn.LSTMCell(reading_rep_dim//2)
            b_read_q_high = tf.contrib.rnn.LSTMCell(reading_rep_dim//2)
            ques_high_h, _ = birnn(cell_fw=f_read_q_high,
                                   cell_bw=b_read_q_high,
                                   inputs=ques_low_h,
                                   dtype=tf.float32,
                                   scope='ques_high_under')
            ques_high_h = tf.concat(ques_high_h, axis=2)

            f_read_p_low = tf.contrib.rnn.LSTMCell(reading_rep_dim//2)
            b_read_p_low = tf.contrib.rnn.LSTMCell(reading_rep_dim//2)
            para_low_h, _ = birnn(cell_fw=f_read_p_low,
                                  cell_bw=b_read_p_low,
                                  inputs=para_enhanced_rep,
                                  dtype=tf.float32,
                                  scope='para_low_under')
            para_low_h = tf.concat(para_low_h, axis=2)

            f_read_p_high = tf.contrib.rnn.LSTMCell(reading_rep_dim//2)
            b_read_p_high = tf.contrib.rnn.LSTMCell(reading_rep_dim//2)
            para_high_h, _ = birnn(cell_fw=f_read_p_high,
                                   cell_bw=b_read_p_high,
                                   inputs=para_low_h,
                                   dtype=tf.float32,
                                   scope='para_high_under')
            para_high_h = tf.concat(para_high_h, axis=2)

        logging.info("Final Question Understanding")

        with tf.variable_scope("final_q_und"):
            f_uq = tf.contrib.rnn.LSTMCell(final_ques_under_dim//2)
            b_uq = tf.contrib.rnn.LSTMCell(final_ques_under_dim//2)
            inp = tf.concat([ques_low_h, ques_high_h], axis=2)
            final_q_und, _ = birnn(cell_fw=f_uq,
                                   cell_bw=b_uq,
                                   inputs=inp,
                                   dtype=tf.float32,
                                   scope='final_q_und')
            final_q_und = tf.concat(final_q_und, axis=2)

        logging.info("Fusion High level")

        with tf.variable_scope("high_level_fusion"):
            para_HoW = tf.concat([para_glove, para_cove,
                                  para_low_h, para_high_h],
                                 axis=2)
            ques_HoW = tf.concat([ques_glove, ques_cove,
                                  ques_low_h, ques_high_h],
                                 axis=2)
            para_fused_l = fuse(para_HoW, ques_HoW, sl_att_dim,
                                B=ques_low_h,
                                scope='low_level_fusion')
            para_fused_h = fuse(para_HoW, ques_HoW, sh_att_dim,
                                B=ques_high_h,
                                scope='high_level_fusion')
            para_fused_u = fuse(para_HoW, ques_HoW, su_att_dim,
                                B=final_q_und,
                                scope='understanding_fusion')
            inp = tf.concat([para_low_h, para_high_h,
                             para_fused_l, para_fused_h,
                             para_fused_u], axis=2)
            f_vc = tf.contrib.rnn.LSTMCell(fully_fused_para_dim//2)
            b_vc = tf.contrib.rnn.LSTMCell(fully_fused_para_dim//2)
            ff_para, _ = birnn(cell_fw=f_vc, cell_bw=b_vc, inputs=inp,
                               dtype=tf.float32, scope='full_fused_para')
            ff_para = tf.concat(ff_para, axis=2)

        logging.info("Self boosting fusion")

        with tf.variable_scope("self_boosting_fusion"):
            para_HoW = tf.concat([para_glove, para_cove,
                                  para_low_h, para_high_h,
                                  para_fused_l, para_fused_h,
                                  para_fused_u, ff_para],
                                 axis=2)
            ff_fused_para = fuse(para_HoW, para_HoW, selfboost_att_dim,
                                 B=ff_para,
                                 scope='self_boosted_fusion')
            f_sb = tf.contrib.rnn.LSTMCell(selfboost_rep_dim//2)
            b_sb = tf.contrib.rnn.LSTMCell(selfboost_rep_dim//2)
            inp = tf.concat([ff_para, ff_fused_para], axis=2)
            final_para_rep, _ = birnn(cell_fw=f_sb, cell_bw=b_sb, inputs=inp,
                                      dtype=tf.float32, scope='self_boosted')
            final_para_rep = tf.concat(final_para_rep, axis=2)

        logging.info("Fusion Net construction complete")
        logging.info("SQuAD specific construction begins")
        # now we have U_c, U_q = final_para_rep, final_q_und
        # The rest of the network is for SQuAD
        # TODO: This part is a little confusing

        logging.info("Sumarized question understanding vector")
        with tf.variable_scope("summarized_question"):
            w = tf.get_variable("W", shape=(final_ques_under_dim, 1),
                                dtype=tf.float32)
            uq_s = tf.unstack(final_q_und, axis=1)
            attention_weight = []
            for i, uq in enumerate(tqdm(uq_s, desc='Question Summary Vector')):
                s = tf.matmul(uq, w)
                attention_weight.append(s)
            attention_weight = tf.nn.softmax(tf.stack(attention_weight,
                                                      axis=1))
            summarized_question = tf.reduce_sum(tf.multiply(final_q_und,
                                                            attention_weight),
                                                axis=1)

        logging.info("Span Start")
        with tf.variable_scope("span_start"):
            w = tf.get_variable("W", shape=(selfboost_rep_dim,
                                            final_ques_under_dim),
                                dtype=tf.float32)
            uc_s = tf.unstack(final_para_rep, axis=1)
            attention_weight = []
            for i, uc in enumerate(tqdm(uc_s, desc='StartSpan')):
                s = tf.matmul(uc, w)
                s = tf.reduce_sum(tf.multiply(s, summarized_question), axis=1)
                attention_weight.append(s)
            start_prediction = tf.nn.softmax(tf.stack(attention_weight,
                                                      axis=1))

        logging.info("Span End")
        with tf.variable_scope("span_end"):
            # final memory of GRU
            inp = tf.multiply(tf.expand_dims(start_prediction, axis=2),
                              final_para_rep)
            sum_dim = summarized_question.get_shape().as_list()[-1]
            out, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(sum_dim),
                                       inputs=inp, dtype=tf.float32,
                                       initial_state=summarized_question,
                                       scope='span_end_question_encoding')
            vq = tf.unstack(out, axis=1)[-1]
            vq_dim = vq.get_shape().as_list()[-1]
            w = tf.get_variable("W", shape=(selfboost_rep_dim, vq_dim),
                                dtype=tf.float32)
            uc_s = tf.unstack(final_para_rep, axis=1)
            attention_weight = []
            for i, uc in enumerate(tqdm(uc_s, desc='StartSpan')):
                s = tf.matmul(uc, w)
                s = tf.reduce_sum(tf.multiply(s, vq), axis=1)
                attention_weight.append(s)
            end_prediction = tf.nn.softmax(tf.stack(attention_weight, axis=1))

        logging.info("Model Creation Complete")
        logging.info("Creating optimizers")
    return (para_glove, ques_glove, para_cove, ques_cove,
            para_nerpos, para_tf, para_em, start_prediction,
            end_prediction)
