class SimpleFeatureExtractor:

    def get_features(self, parser_state, **kwargs):
       
        stack_len = 2
        input_buffer_len = 1
        stack_items = parser_state.stack_peek_n(stack_len)
        input_buffer_items = parser_state.input_buffer_peek_n(input_buffer_len)
        features = []
        assert len(stack_items) == stack_len
        assert len(input_buffer_items) == input_buffer_len
        features.extend([x.embedding for x in stack_items])
        features.extend([x.embedding for x in input_buffer_items])
        return features
