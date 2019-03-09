from collections import namedtuple

from ..config import Actions, NULL_STACK_TOK

DepGraphEdge = namedtuple("DepGraphEdge", ["head", "modifier"])

StackEntry = namedtuple("StackEntry", ["headword", "headword_pos", "embedding"])


class ParserState:

    def __init__(self, sentence, sentence_embs, combiner, null_stack_tok_embed=None):
        self.combiner = combiner
        self.curr_input_buff_idx = 0
        self.input_buffer = [StackEntry(we[0], pos, we[1]) for pos, we in enumerate(zip(sentence, sentence_embs))]

        self.stack = []
        self.null_stack_tok_embed = null_stack_tok_embed

    def shift(self):
        next_item = self.input_buffer[self.curr_input_buff_idx]
        self.stack.append(next_item)
        self.curr_input_buff_idx += 1

    def reduce_left(self):
        return self._reduce(Actions.REDUCE_L)

    def reduce_right(self):
        return self._reduce(Actions.REDUCE_R)

    def done_parsing(self):
        if len(self.stack) == 1 and self.curr_input_buff_idx == len(self.input_buffer) - 1:
            return True
        else:
            return False

    def stack_len(self):
        return len(self.stack)

    def input_buffer_len(self):
        return len(self.input_buffer) - self.curr_input_buff_idx

    def stack_peek_n(self, n):
        if len(self.stack) - n < 0:
            return [StackEntry(NULL_STACK_TOK, -1, self.null_stack_tok_embed)] * (n - len(self.stack)) \
                   + self.stack[:]
        return self.stack[-n:]

    def input_buffer_peek_n(self, n):
        assert self.curr_input_buff_idx + n - 1 <= len(self.input_buffer)
        return self.input_buffer[self.curr_input_buff_idx:self.curr_input_buff_idx+n]

    def _reduce(self, action):
        assert len(self.stack) >= 2, "ERROR: Cannot reduce with stack length less than 2"
        
        if action == Actions.REDUCE_L:
            head = self.stack.pop()
            modifier = self.stack.pop()
        elif action == Actions.REDUCE_R:
            modifier = self.stack.pop()
            head = self.stack.pop()
        head_embedding = self.combiner(head.embedding, modifier.embedding)
        self.stack.append(StackEntry(head.headword, head.headword_pos, head_embedding))
        return DepGraphEdge((head.headword, head.headword_pos), (modifier.headword, modifier.headword_pos))

    def __str__(self):
        return "Stack: {}\nInput Buffer: {}\n".format([entry.headword for entry in self.stack],
                                                      [entry.headword for entry
                                                       in self.input_buffer[self.curr_input_buff_idx:]])
