import re
from collections import defaultdict
import tensorflow as tf
import numpy as np
from metrics import precision_recall_f1


DATA_ROOT='./data/ner'
URL_RE = re.compile('(https?)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]')
NICKNAME_RE = re.compile('@[a-zA-Z]+')
URL_TOK = '<URL>'
USER_TOK = '<USR>'


def read_data(file_path):
    tokens = []
    tags = []

    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            # Replace all urls with <URL> token
            # Replace all users with <USR> token
            token = re.sub(URL_RE, URL_TOK, token)
            token = re.sub(NICKNAME_RE, USER_TOK, token)

            tweet_tokens.append(token)
            tweet_tags.append(tag)

    return tokens, tags


def build_dict(tokens_or_tags, special_tokens):
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
    """

    # Create mappings from tokens (or tags) to indices and vice versa.

    tok2idx = defaultdict(lambda: 0)
    idx2tok = []
    i = 0;

    for sp in special_tokens:
        tok2idx[sp] = i
        idx2tok.append(sp)
        i = i + 1

    words = []
    for line in tokens_or_tags:
        words = words + line

    vocab = set(words)

    for word in vocab:
        tok2idx[word] = i
        idx2tok.append(word)
        i = i + 1

    # tok2idx = dict([special_tokens[i],i] for i in range(len(special_tokens)))
    # idx2tok = []
    # idx2tok.extend(special_tokens)
    # i = len(tok2idx)
    # for list_tok_tag in tokens_or_tags:
    #     for tok_tag in list_tok_tag:
    #         if tok_tag in tok2idx:
    #             continue
    #         tok2idx[tok_tag] = i
    #         idx2tok.append(tok_tag)
    #         i +=1

    return tok2idx, idx2tok

def words2idxs(tokens_list):
    return [token2idx[word] for word in tokens_list]

def tags2idxs(tags_list):
    return [tag2idx[tag] for tag in tags_list]

def idxs2words(idxs):
    return [idx2token[idx] for idx in idxs]

def idxs2tags(idxs):
    return [idx2tag[idx] for idx in idxs]


def batches_generator(batch_size, tokens, tags,
                      shuffle=True, allow_smaller_last_batch=True):
    """Generates padded batches of tokens and tags."""

    n_samples = len(tokens)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    n_batches = n_samples // batch_size
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list = []
        y_list = []
        max_len_token = 0
        for idx in order[batch_start: batch_end]:
            x_list.append(words2idxs(tokens[idx]))
            y_list.append(tags2idxs(tags[idx]))
            max_len_token = max(max_len_token, len(tags[idx]))

        # Fill in the data into numpy nd-arrays filled with padding indices.
        x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * token2idx['<PAD>']
        y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * tag2idx['O']
        lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            utt_len = len(x_list[n])
            x[n, :utt_len] = x_list[n]
            lengths[n] = utt_len
            y[n, :utt_len] = y_list[n]
        yield x, y, lengths

def predict_tags(model, session, token_idxs_batch, lengths):
    """Performs predictions and transforms indices to tokens and tags."""

    tag_idxs_batch = model.predict_for_batch(session, token_idxs_batch, lengths)

    tags_batch, tokens_batch = [], []
    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
        tags, tokens = [], []
        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
            tags.append(idx2tag[tag_idx])
            tokens.append(idx2token[token_idx])
        tags_batch.append(tags)
        tokens_batch.append(tokens)
    return tags_batch, tokens_batch


def eval_conll(model, session, tokens, tags, short_report=True):
    """Computes NER quality measures using CONLL shared task script."""

    y_true, y_pred = [], []
    for x_batch, y_batch, lengths in batches_generator(1, tokens, tags):
        tags_batch, tokens_batch = predict_tags(model, session, x_batch, lengths)
        if len(x_batch[0]) != len(tags_batch[0]):
            raise Exception("Incorrect length of prediction for the input, "
                            "expected length: %i, got: %i" % (len(x_batch[0]), len(tags_batch[0])))
        predicted_tags = []
        ground_truth_tags = []
        for gt_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]):
            if token != '<PAD>':
                ground_truth_tags.append(idx2tag[gt_tag_idx])
                predicted_tags.append(pred_tag)

        # We extend every prediction and ground truth sequence with 'O' tag
        # to indicate a possible end of entity.
        y_true.extend(ground_truth_tags + ['O'])
        y_pred.extend(predicted_tags + ['O'])

    results = precision_recall_f1(y_true, y_pred, print_results=True, short_report=short_report)
    return results


class BiLSTMModel:

    def __init__(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
        self.init_model(vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index)

    @classmethod
    def declare_placeholders(self):
        """Specifies placeholders for the model."""

        # Placeholders for input and ground truth output.
        self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch')
        self.ground_truth_tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ground_truth_tags')

        # Placeholder for lengths of the sequences.
        self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')

        # Placeholder for a dropout keep probability. If we don't feed
        # a value for this placeholder, it will be equal to 1.0.
        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])

        # Placeholder for a learning rate (tf.float32).
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_ph')

    @classmethod
    def build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
        """Specifies bi-LSTM architecture and computes logits for inputs."""

        # Create embedding variable
        initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
        embedding_matrix_variable =   tf.Variable(initial_value=initial_embedding_matrix, name='embedding_matrix', dtype=tf.float32)

        # Create RNN cells and dropout, initializing all *_keep_prob with dropout placeholder.
        forward_cell =  tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_hidden_rnn), input_keep_prob=self.dropout_ph,
                                                      output_keep_prob=self.dropout_ph, state_keep_prob=self.dropout_ph)
        backward_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_hidden_rnn), input_keep_prob=self.dropout_ph,
                                                      output_keep_prob=self.dropout_ph, state_keep_prob=self.dropout_ph)

        # Look up embeddings for self.input_batch.
        # Shape: [batch_size, sequence_len, embedding_dim].
        embeddings =   tf.nn.embedding_lookup(embedding_matrix_variable,self.input_batch)

        # Pass them through Bidirectional Dynamic RNN
        # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn].
        (rnn_output_fw, rnn_output_bw), _ =  tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, embeddings, sequence_length=self.lengths, dtype=tf.float32)
        rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

        # Dense layer on top.
        # Shape: [batch_size, sequence_len, n_tags].
        self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)

    @classmethod
    def compute_predictions(self):
        """Transforms logits to probabilities and finds the most probable tags."""

        # Create softmax function
        softmax_output = tf.nn.softmax(self.logits)

        # Use argmax (to get the most probable tags
        self.predictions = tf.argmax(softmax_output, axis=-1, name='predictions')

    @classmethod
    def compute_loss(self, n_tags, PAD_index):
        """Computes masked cross-entopy loss with logits."""

        # Create cross entropy function function
        ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth_tags_one_hot, logits=self.logits)

        mask = tf.cast(tf.not_equal(self.input_batch, PAD_index), tf.float32)
        # Create loss function which doesn't operate with <PAD> tokens
        self.loss = tf.reduce_mean(mask*loss_tensor)


    @classmethod
    def perform_optimization(self):
        """Specifies the optimizer and train_op for the model."""

        self.optimizer =  tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

        # Gradient clipping
        clip_norm = tf.cast(1.0, tf.float32)
        self.grads_and_vars = [(tf.clip_by_norm(gradient, clip_norm), var) for gradient, var in self.grads_and_vars]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

    @classmethod
    def init_model(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
        self.declare_placeholders()
        self.build_layers(vocabulary_size, embedding_dim, n_hidden_rnn, n_tags)
        self.compute_predictions()
        self.compute_loss(n_tags, PAD_index)
        self.perform_optimization()

    @classmethod
    def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability):
        feed_dict = {self.input_batch: x_batch,
                     self.ground_truth_tags: y_batch,
                     self.learning_rate_ph: learning_rate,
                     self.dropout_ph: dropout_keep_probability,
                     self.lengths: lengths}

        session.run(self.train_op, feed_dict=feed_dict)

    @classmethod
    def predict_for_batch(self, session, x_batch, lengths):
        predictions =session.run(self.predictions, feed_dict={self.input_batch: x_batch, self.lengths: lengths})
        return predictions



if __name__ == "__main__":

    train_tokens, train_tags = read_data(DATA_ROOT + '/train.txt')
    validation_tokens, validation_tags = read_data(DATA_ROOT + '/validation.txt')
    test_tokens, test_tags = read_data(DATA_ROOT + '/test.txt')

    # for i in range(3):
    #     for token, tag in zip(train_tokens[i], train_tags[i]):
    #         print('%s\t%s' % (token, tag))
    #     print()

    special_tokens = ['<UNK>', '<PAD>']
    special_tags = ['O']

    # Create dictionaries
    token2idx, idx2token = build_dict(train_tokens + validation_tokens, special_tokens)
    print("len(token2idx)={}, len(idx2token)={}".format(len(token2idx), len(idx2token)))
    tag2idx, idx2tag = build_dict(train_tags, special_tags)
    print("len(tag2idx)={}, len(idx2tag)={}".format(len(tag2idx), len(idx2tag)))

    tf.reset_default_graph()

    model = BiLSTMModel(vocabulary_size=len(token2idx), n_tags=len(tag2idx), embedding_dim=200, n_hidden_rnn=200, PAD_index=token2idx['<PAD>'])

    batch_size =  32
    n_epochs =  4
    learning_rate =  0.005
    learning_rate_decay =  np.sqrt(2);
    # 0.1, 0.5, 0.6, 0.9.
    dropout_keep_probability =  0.6

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Start training... \n')
    for epoch in range(n_epochs):
        # For each epoch evaluate the model on train and validation data
        print('-' * 20 + ' Epoch {} '.format(epoch + 1) + 'of {} '.format(n_epochs) + '-' * 20)
        print('Train data evaluation:')
        eval_conll(model, sess, train_tokens, train_tags, short_report=True)
        print('Validation data evaluation:')
        eval_conll(model, sess, validation_tokens, validation_tags, short_report=True)

        # Train the model
        for x_batch, y_batch, lengths in batches_generator(batch_size, train_tokens, train_tags):
            model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)

        # Decaying the learning rate
        learning_rate = learning_rate / learning_rate_decay

    print('...training finished.')

    print('-' * 20 + ' Train set quality: ' + '-' * 20)
    train_results = eval_conll(model, sess, train_tokens, train_tags, short_report=False)

    print('-' * 20 + ' Validation set quality: ' + '-' * 20)
    validation_results = eval_conll(model, sess, validation_tokens, validation_tags, short_report=False)

    print('-' * 20 + ' Test set quality: ' + '-' * 20)
    test_results = eval_conll(model, sess, test_tokens, test_tags, short_report=False)