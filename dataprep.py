def batchgen(workspace_path):
    "Read batches from file from the given workspace path"
    for i in range(1):
        yield dict(para_glove=None,
                   ques_glove=None,
                   para_cove=None,
                   ques_cove=None,
                   para_mask=None,
                   ques_mask=None,
                   para_nerpos=None,
                   para_tf=None,
                   para_em=None,
                   ans_start=None,
                   ans_end=None)
