import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

word2id = {symbol:i for i, symbol in enumerate('^$#+-1234567890')}
id2word = {i:symbol for symbol, i in word2id.items()}
start_symbol = '^'
end_symbol = '$'
padding_symbol = '#'

def generate_equations(allowed_operators, dataset_size, min_value, max_value):
    """Generates pairs of equations and solutions to them.

       Each equation has a form of two integers with an operator in between.
       Each solution is an integer with the result of the operaion.

        allowed_operators: list of strings, allowed operators.
        dataset_size: an integer, number of equations to be generated.
        min_value: an integer, min value of each operand.
        max_value: an integer, max value of each operand.

        result: a list of tuples of strings (equation, solution).
    """

    sample = []
    for _ in range(dataset_size):
        op = random.choice(allowed_operators)
        n1 = random.randint(min_value, max_value)
        n2 = random.randint(min_value, max_value)
        equation = '{}{}{}'.format(n1,op,n2)
        solution = str(eval(equation))
        sample.append((equation, solution))
    return sample


def sentence_to_ids(sentence, word2id, padded_len):
    """ Converts a sequence of symbols to a padded sequence of their ids.

      sentence: a string, input/output sequence of symbols.
      word2id: a dict, a mapping from original symbols to ids.
      padded_len: an integer, a desirable length of the sequence.

      result: a tuple of (a list of ids, an actual length of sentence).
    """
    padded_sen = (sentence[:padded_len - 1] + end_symbol).ljust(padded_len, padding_symbol)
    sent_len = padded_sen.find(padding_symbol)
    if sent_len == -1:
        sent_len = len(padded_sen)
    sent_ids = [word2id[i] for i in list(padded_sen)]
    return sent_ids, sent_len


def ids_to_sentence(ids, id2word):
    """ Converts a sequence of ids to a sequence of symbols.

          ids: a list, indices for the padded sequence.
          id2word:  a dict, a mapping from ids to original symbols.

          result: a list of symbols.
    """

    return [id2word[i] for i in ids]


def batch_to_ids(sentences, word2id, max_len):
    """Prepares batches of indices.

       Sequences are padded to match the longest sequence in the batch,
       if it's longer than max_len, then max_len is used instead.

        sentences: a list of strings, original sequences.
        word2id: a dict, a mapping from original symbols to ids.
        max_len: an integer, max len of sequences allowed.

        result: a list of lists of ids, a list of actual lengths.
    """

    max_len_in_batch = min(max(len(s) for s in sentences) + 1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len

def generate_batches(samples, batch_size=64):
    X, Y = [], []
    for i, (x, y) in enumerate(samples, 1):
        X.append(x)
        Y.append(y)
        if i % batch_size == 0:
            yield X, Y
            X, Y = [], []
    if X and Y:
        yield X, Y



class Seq2SeqModel:


    def __init__(self, vocab_size, embeddings_size, hidden_size, max_iter, start_symbol_id, end_symbol_id, padding_symbol_id):
        self.init_model(vocab_size, embeddings_size, hidden_size, max_iter, start_symbol_id, end_symbol_id, padding_symbol_id)

    @classmethod
    def declare_placeholders(self):
        """Specifies placeholders for the model."""

        # Placeholders for input and its actual lengths.
        self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_batch')
        self.input_batch_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_batch_lengths')

        # Placeholders for groundtruth and its actual lengths.
        self.ground_truth = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ground_truth')
        self.ground_truth_lengths = tf.placeholder(shape=(None, ), dtype=tf.int32, name='ground_truth_lengths')

        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_ph')

    @classmethod
    def create_embeddings(self, vocab_size, embeddings_size):
        """Specifies embeddings layer and embeds an input batch."""

        random_initializer = tf.random_uniform((vocab_size, embeddings_size), -1.0, 1.0)
        # Create embedding variable
        self.embeddings = tf.Variable(initial_value=random_initializer, dtype=tf.float32, name='embeddings')

        # Perform embeddings lookup for self.input_batch.
        self.input_batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_batch)

    @classmethod
    def build_encoder(self, hidden_size):
        """Specifies encoder architecture and computes its output."""

        # Create GRUCell with dropout.
        encoder_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size),
                                                     input_keep_prob=self.dropout_ph,  dtype = tf.float32)
        #                                             output_keep_prob=self.dropout_ph, state_keep_prob=self.dropout_ph)

        # Create RNN with the predefined cell.
        _, self.final_encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.input_batch_embedded,
                                                        sequence_length=self.input_batch_lengths,
                                                        dtype=tf.float32)

    @classmethod
    def build_decoder(self, hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id):
        """Specifies decoder architecture and computes the output.

            Uses different helpers:
              - for train: feeding ground truth
              - for inference: feeding generated output

            As a result, self.train_outputs and self.infer_outputs are created.
            Each of them contains two fields:
              rnn_output (predicted logits)
              sample_id (predictions).

        """

        # Use start symbols as the decoder inputs at the first time step.
        batch_size = tf.shape(self.input_batch)[0]
        start_tokens = tf.fill([batch_size], start_symbol_id)
        ground_truth_as_input = tf.concat([tf.expand_dims(start_tokens, 1), self.ground_truth], 1)

        # Use the embedding layer defined before to lookup embedings for ground_truth_as_input.
        self.ground_truth_embedded = tf.nn.embedding_lookup(params=self.embeddings, ids=ground_truth_as_input)

        # Create TrainingHelper for the train stage.
        train_helper = tf.contrib.seq2seq.TrainingHelper(self.ground_truth_embedded,
                                                         self.ground_truth_lengths)

        # Create GreedyEmbeddingHelper for the inference stage.
        # You should provide the embedding layer, start_tokens and index of the end symbol.
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                start_tokens=start_tokens,
                                                                end_token=end_symbol_id)

        def decode(helper, scope, reuse=None):
            """Creates decoder and return the results of the decoding with a given helper."""

            with tf.variable_scope(scope, reuse=reuse):
                # Create GRUCell with dropout. Do not forget to set the reuse flag properly.
                decoder_cell = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.GRUCell(hidden_size, reuse=reuse, name='decoder_cell'),
                    input_keep_prob=self.dropout_ph, dtype = tf.float32)

                # Create a projection wrapper.
                decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, vocab_size, reuse=reuse)

                # Create BasicDecoder, pass the defined cell, a helper, and initial state.
                # The initial state should be equal to the final state of the encoder!
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, self.final_encoder_state)

                # The first returning argument of dynamic_decode contains two fields:
                #   rnn_output (predicted logits)
                #   sample_id (predictions)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=max_iter,
                                                                  output_time_major=False, impute_finished=True)

                return outputs

        self.train_outputs = decode(train_helper, 'decode')
        self.infer_outputs = decode(infer_helper, 'decode', reuse=True)

    @classmethod
    def compute_loss(self):
        """Computes sequence loss (masked cross-entopy loss with logits)."""

        weights = tf.cast(tf.sequence_mask(self.ground_truth_lengths), dtype=tf.float32)

        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.train_outputs.rnn_output, targets=self.ground_truth, weights=weights)

    @classmethod
    def perform_optimization(self):
        """Specifies train_op that optimizes self.loss."""
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                global_step=tf.train.get_global_step(),
                                                learning_rate=self.learning_rate_ph,
                                                optimizer='Adam',
                                                clip_gradients=1.0)

    @classmethod
    def init_model(self, vocab_size, embeddings_size, hidden_size,
                   max_iter, start_symbol_id, end_symbol_id, padding_symbol_id):
        self.declare_placeholders()
        self.create_embeddings(vocab_size, embeddings_size)
        self.build_encoder(hidden_size)
        self.build_decoder(hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id)

        # Compute loss and back-propagate.
        self.compute_loss()
        self.perform_optimization()

        # Get predictions for evaluation.
        self.train_predictions = self.train_outputs.sample_id
        self.infer_predictions = self.infer_outputs.sample_id

    @classmethod
    def train_on_batch(self, session, X, X_seq_len, Y, Y_seq_len, learning_rate, dropout_keep_probability):
        feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len,
            self.learning_rate_ph: learning_rate,
            self.dropout_ph: dropout_keep_probability
        }
        pred, loss, _ = session.run([
            self.train_predictions,
            self.loss,
            self.train_op], feed_dict=feed_dict)
        return pred, loss

    @classmethod
    def predict_for_batch(self, session, X, X_seq_len):
        feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len
        }

        pred = session.run([
            self.infer_predictions
        ], feed_dict=feed_dict)[0]
        return pred

    @classmethod
    def predict_for_batch_with_loss(self, session, X, X_seq_len, Y, Y_seq_len):
        feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len,
        }
        pred, loss = session.run([
            self.infer_predictions,
            self.loss,
        ], feed_dict=feed_dict)
        return pred, loss


if __name__ == "__main__":
    allowed_operators = ['+', '-']
    dataset_size = 100000
    data = generate_equations(allowed_operators, dataset_size, min_value=0, max_value=9999)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    tf.reset_default_graph()
    model = Seq2SeqModel(vocab_size=len(word2id), embeddings_size=20, hidden_size=512,
                         max_iter=7, start_symbol_id=word2id[start_symbol], end_symbol_id=word2id[end_symbol], padding_symbol_id=word2id[padding_symbol])
    batch_size = 128
    n_epochs = 15
    #0.001
    learning_rate = 0.001
    # 0.5
    dropout_keep_probability = 0.5
    max_len = 20

    n_step = int(len(train_set) / batch_size)

    my_config = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=5, \
                               allow_soft_placement=True, device_count={'CPU': 2})
    session = tf.Session(config=my_config)
    session.run(tf.global_variables_initializer())

    #session = tf.Session()
    #session.run(tf.global_variables_initializer())

    invalid_number_prediction_counts = []
    all_model_predictions = []
    all_ground_truth = []

    print('Start training... \n')
    for epoch in range(n_epochs):
        random.shuffle(train_set)
        random.shuffle(test_set)

        print('Train: epoch', epoch + 1)
        for n_iter, (X_batch, Y_batch) in enumerate(generate_batches(train_set, batch_size=batch_size)):
            # prepare the data (X_batch and Y_batch) for training
            # using function batch_to_ids
            X_ids, X_sent_lens = batch_to_ids(X_batch, word2id, max_len=max_len)
            Y_ids, Y_sent_lens = batch_to_ids(Y_batch, word2id, max_len=max_len)
            predictions, loss =  model.train_on_batch(session, X_ids, X_sent_lens, Y_ids, Y_sent_lens,learning_rate,
            dropout_keep_probability)

            if n_iter % 200 == 0:
                print("Epoch: [%d/%d], step: [%d/%d], loss: %f" % (epoch + 1, n_epochs, n_iter + 1, n_step, loss))

        X_sent, Y_sent = next(generate_batches(test_set, batch_size=batch_size))
        # prepare test data (X_sent and Y_sent) for predicting
        # quality and computing value of the loss function
        # using function batch_to_ids
        X, X_sent_lens = batch_to_ids(X_sent, word2id, max_len=max_len)
        Y, Y_sent_lens = batch_to_ids(Y_sent, word2id, max_len=max_len)
        predictions, loss = model.predict_for_batch_with_loss(session, X, X_sent_lens, Y, Y_sent_lens)
        print('Test: epoch', epoch + 1, 'loss:', loss, )
        for x, y, p in list(zip(X, Y, predictions))[:3]:
            print('X:', ''.join(ids_to_sentence(x, id2word)))
            print('Y:', ''.join(ids_to_sentence(y, id2word)))
            print('O:', ''.join(ids_to_sentence(p, id2word)))
            print('')

        model_predictions = []
        ground_truth = []
        invalid_number_prediction_count = 0
        # For the whole test set calculate ground-truth values (as integer numbers)
        # and prediction values (also as integers) to calculate metrics.
        # If generated by model number is not correct (e.g. '1-1'),
        # increase invalid_number_prediction_count and don't append this and corresponding
        # ground-truth value to the arrays.
        for X_batch, Y_batch in generate_batches(test_set, batch_size=batch_size):
            X_ids, X_sent_lens = batch_to_ids(X_batch, word2id, max_len=max_len)
            Y_ids, Y_sent_lens = batch_to_ids(Y_batch, word2id, max_len=max_len)
            predictions = model.predict_for_batch(session, X_ids, X_sent_lens)
            for y, p in zip(Y_ids, predictions):
                y_sent = ''.join(ids_to_sentence(y, id2word))
                y_sent = y_sent[:y_sent.find('$')]
                p_sent = ''.join(ids_to_sentence(p, id2word))
                p_sent = p_sent[:p_sent.find('$')]
                if p_sent.isdigit() or (p_sent.startswith('-') and p_sent[1:].isdigit()):
                    model_predictions.append(int(p_sent))
                    ground_truth.append(int(y_sent))
                else:
                    invalid_number_prediction_count += 1

        all_model_predictions.append(model_predictions)
        all_ground_truth.append(ground_truth)
        invalid_number_prediction_counts.append(invalid_number_prediction_count)

    print('\n...training finished.')
    for i, (gts, predictions, invalid_number_prediction_count) in enumerate(zip(all_ground_truth,
                                                                                all_model_predictions,
                                                                                invalid_number_prediction_counts), 1):
        mae = mean_absolute_error(gts, predictions)
        print("Epoch: %i, MAE: %f, Invalid numbers: %i" % (i, mae, invalid_number_prediction_count))