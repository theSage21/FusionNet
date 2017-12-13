import spacy
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool


nlp = spacy.load('en')
# NOTE: Vocab list maps words to indices
glove_map = {}


def embed_nerpos(w):
    pos = w.pos  # some number
    # TODO: NER to be done
    return pos


def process_para(p, max_p_len):
    pp = nlp(p)  # Parsed Para
    pad_len = (max_p_len - len(pp))
    para_mask = ([1] * len(pp)) + ([0] * pad_len)
    para_glove = [glove_map.get(w.text, 1) for w in pp] + ([0] * pad_len)
    para_nerpos = [embed_nerpos(w) for w in pp] + ([0]*pad_len)
    w_counts = Counter([w.text for w in pp])
    para_tf = [w_counts[w.text] for w in pp] + ([0]*pad_len)
    return para_glove, para_mask, para_nerpos, para_tf


def process_question(q, max_q_len):
    pq = nlp.tokenizer(q)  # Parsed Question
    pad_len = (max_q_len - len(pq))
    ques_mask = ([1] * len(pq)) + ([0] * pad_len)
    ques_glove = [glove_map.get(w.text, 1) for w in pq] + ([0]*pad_len)
    return ques_glove, ques_mask


def __parallel_worker(args):
    "Lets you run arbitrary functions in parallel using the imap_unordered fn"
    fn, *actual_args = args
    retval = fn(*actual_args)
    return retval


def populate_workspace(workspace_path, squadfile, max_p_len, max_q_len):
    "Populates the workspace"
    to_process = []
    for doc in tqdm(squadfile['data'], desc='Document'):
        for para in tqdm(doc['paragraphs'], desc='Para'):
            context = para['context']
            for qas in tqdm(para['qas'], desc='Question'):
                question = qas['question']
                answer = qas['answers']
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer['text'])
                to_process.append((qas['id'], context, question,
                                   answer_start, answer_end))
    # TODO: Process these items in parallel
    with Pool() as pool:
        paras = list(set([i[1] for i in to_process]))
        arguments = [(process_para, (p, max_p_len))
                     for p in paras]
        work = pool.imap_unordered(__parallel_worker, arguments)
        for item in tqdm(work, total=len(arguments),
                         desc='Para procecssing'):
            pass
        # -------question processing
        questions = list(set([i[2] for i in to_process]))
        arguments = [(process_question, (q, max_q_len))
                     for q in questions]
        work = pool.imap_unordered(__parallel_worker, arguments)
        for item in tqdm(work, total=len(arguments),
                         desc='Question procecssing'):
            pass


def batchgen(workspace_path):
    "Read batches from file from the given workspace path"
    # NOTE: We generate para_em here
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
