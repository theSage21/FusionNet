import logging
import tensorflow as tf
from model import build
from config import config
from dataprep import batchgen
from tqdm import tqdm, trange


config['is_train'] = True


(para_glove, ques_glove, para_cove, ques_cove,
 para_nerpos, para_tf, para_em, start_prediction,
 end_prediction, exp_ans_start, exp_ans_end,
 inp_para_mask, inp_ques_mask) = build(**config)

logging.info("Creating loss")

# NOTE: The paper says maximizing the sum of sum of log probabilities.
# I'm not sure what that means. Sticking with the loss I understand
lossfunc = tf.nn.sigmoid_cross_entropy_with_logits
start_loss = lossfunc(labels=exp_ans_start,
                      logits=start_prediction)
end_loss = lossfunc(labels=exp_ans_end,
                    logits=end_prediction)

total_loss = start_loss + end_loss

logging.info("Calculating gradients for optimizer")
opt = tf.train.AdagradOptimizer(learning_rate=config['learning_rate'])
minimization = opt.minimize(total_loss)
logging.info("Beginning Training")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # --------------
    for epoch in trange(config['n_epochs']):
        loss_seen = []
        for b in tqdm(batchgen(config['workspace_location']),
                      desc='Batch'):
            feed = {para_glove: b['para_glove'],
                    ques_glove: b['ques_glove'],
                    para_cove: b['para_cove'],
                    ques_cove: b['ques_cove'],
                    para_nerpos: b['para_nerpos'],
                    para_tf: b['para_tf'],
                    para_em: b['para_em'],
                    exp_ans_start: b['ans_start'],
                    exp_ans_end: b['ans_end'],
                    inp_para_mask: b['para_mask'],
                    inp_ques_mask: b['ques_mask']}
            to_run = [total_loss, minimization]
            sess.run(to_run, feed_dict=feed)
