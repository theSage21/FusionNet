batchsize = 32
nerpos_dim = 20  # in paper 20
tf_dim = 1  # in paper 1
max_p_len = 15
max_q_len = 14
glove_dim = 300  # in paper 300
cove_dim = 600  # in paper 600
reading_rep_dim = 50  # in paper 250
final_ques_under_dim = 50  # in paper 250
fully_fused_para_dim = 50  # in paper 250

sl_att_dim = 13  # NOTE unable to find in paper
sh_att_dim = 12  # NOTE unable to find in paper
su_att_dim = 11  # NOTE unable to find in paper

selfboost_att_dim = 10  # NOTE unable to find in paper
selfboost_rep_dim = 8  # in paper 250

learning_rate_alpha = 0.002  # as in paper
learning_rate_beta = (0.9, 0.999)  # as in paper
epsilon = 1e-8  # as in paper
n_epochs = 50  # NOTE unable to find in paper


n_models_for_ensemble = 31  # as in paper
