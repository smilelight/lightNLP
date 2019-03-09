import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vectors

from ...utils.log import logger
from ...base.model import BaseConfig, BaseModel

from .config import DEVICE, DEFAULT_CONFIG

from collections import deque

from .config import Actions, NULL_STACK_TOK, END_OF_INPUT_TOK, ROOT_TOK
from .tool import TEXT

from .utils.parser_state import ParserState, DepGraphEdge
from .utils import vectors
from .utils.feature_extractor import SimpleFeatureExtractor
from .components.combiner import MLPCombinerNetwork, LSTMCombinerNetwork
from .components.action_chooser import ActionChooserNetwork
from .components.word_embedding import VanillaWordEmbeddingLookup, BiLSTMWordEmbeddingLookup


class Config(BaseConfig):
    def __init__(self, word_vocab, action_vocab, vector_path, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.action_vocab = action_vocab
        self.action_num = len(self.action_vocab)
        self.vocabulary_size = len(self.word_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class TransitionParser(BaseModel):

    def __init__(self, args):
        super(TransitionParser, self).__init__(args)

        self.args = args
        self.action_num = args.action_num
        self.batch_size = args.batch_size
        self.save_path = args.save_path

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        # feature extractor
        self.feature_extractor = SimpleFeatureExtractor() if args.feature_extractor == 'default' else None

        if args.embedding_type == 'lstm':
            self.word_embedding_component = BiLSTMWordEmbeddingLookup(vocabulary_size, args.word_embedding_dim,
                                                                      args.stack_embedding_dim, args.embedding_lstm_layers,
                                                                      args.embedding_lstm_dropout, args.vector_path,
                                                                      args.non_static)
        elif args.embedding_type == 'vanilla':
            self.word_embedding_component = VanillaWordEmbeddingLookup(vocabulary_size, args.word_embedding_dim)
        else:
            self.word_embedding_component = None
        
        self.action_chooser = ActionChooserNetwork(args.stack_embedding_dim * args.num_features) if args.action_chooser == 'default' else None

        if args.combiner == 'lstm':
            self.combiner = LSTMCombinerNetwork(args.stack_embedding_dim, args.combiner_lstm_layers, args.combiner_lstm_dropout)
        elif args.combiner == 'mlp':
            self.combiner = MLPCombinerNetwork(args.stack_embedding_dim)
        else:
            self.combiner = None

        self.null_stack_tok_embed = torch.randn(1, self.word_embedding_component.output_dim).to(DEVICE)

    def forward(self, sentence, actions=None):
        self.refresh()  # clear up hidden states from last run, if need be

        # Initialize the parser state
        sentence_embs = self.word_embedding_component(sentence)

        parser_state = ParserState(sentence, sentence_embs, self.combiner,
                                   null_stack_tok_embed=self.null_stack_tok_embed)
        outputs = []  # Holds the output of each action decision
        actions_done = []  # Holds all actions we have done

        dep_graph = set()  # Build this up as you go

        # Make the action queue if we have it
        if actions is not None:
            action_queue = deque()
            action_queue.extend([Actions.action_to_ix[a] for a in actions])
            have_gold_actions = True
        else:
            have_gold_actions = False

        while True:
            if parser_state.done_parsing():
                break
            # get features
            features = self.feature_extractor.get_features(parser_state)

            # get log probabilities over actions
            log_probs = self.action_chooser(features)

            # get next step action
            if have_gold_actions:
                temp_action = action_queue.popleft()
            else:
                temp_action = vectors.argmax(log_probs)
            
            # rectify action
            if parser_state.input_buffer_len() == 1:
                temp_action = Actions.REDUCE_R
            elif parser_state.stack_len() < 2:
                temp_action = Actions.SHIFT

            # update parser_state
            if temp_action == Actions.SHIFT:
                parser_state.shift()
                reduction = None
            elif temp_action == Actions.REDUCE_L:
                reduction = parser_state.reduce_left()
            elif temp_action == Actions.REDUCE_R:
                reduction = parser_state.reduce_right()
            else:
                raise Exception('unvalid action!: {}'.format(temp_action))
            
            # keep track of states
            outputs.append(log_probs)
            if reduction:
                dep_graph.add(reduction)
            actions_done.append(temp_action)

        dep_graph.add(DepGraphEdge((TEXT.vocab.stoi[ROOT_TOK], -1), (parser_state.stack[-1].headword, parser_state.stack[-1].headword_pos)))
        return outputs, dep_graph, actions_done

    def refresh(self):
        if isinstance(self.combiner, LSTMCombinerNetwork):
            self.combiner.clear_hidden_state()
        if isinstance(self.word_embedding_component, BiLSTMWordEmbeddingLookup):
            self.word_embedding_component.clear_hidden_state()

    def predict(self, sentence):
        _, dep_graph, _ = self.forward(sentence)
        return dep_graph

    def predict_actions(self, sentence):
        _, _, actions_done = self.forward(sentence)
        return actions_done

    def to_cuda(self):
        self.word_embedding_component.use_cuda = True
        self.combiner.use_cuda = True
        self.cuda()

    def to_cpu(self):
        self.word_embedding_component.use_cuda = False
        self.combiner.use_cuda = False
        self.cpu()

