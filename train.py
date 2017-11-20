from model import build
from config import config


config['is_train'] = True

(para_glove, ques_glove, para_cove, ques_cove,
 para_nerpos, para_tf, para_em, start_prediction,
 end_prediction) = build(**config)
