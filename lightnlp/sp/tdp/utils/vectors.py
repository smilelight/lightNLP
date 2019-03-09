import torch
import torch.autograd as ag

from ..config import DEVICE

def word_to_variable_embed(word, word_embeddings, word_to_ix):
    return word_embeddings(torch.tensor([word_to_ix[word]]).to(DEVICE))


def sequence_to_variable(sequence, to_ix):
    return torch.tensor([to_ix[t] for t in sequence]).to(DEVICE)


def to_scalar(var):
    """
    Wrap up the terse, obnoxious code to go from torch.Tensor to
    a python int / float (is there a better way?)
    """
    if isinstance(var, ag.Variable):
        return var.data.view(-1).tolist()[0]
    else:
        return var.view(-1).tolist()[0]


def argmax(vector):
    """
    Takes in a row vector (1xn) and returns its argmax
    """
    _, idx = torch.max(vector, 1)
    return to_scalar(idx)


def concat_and_flatten(items):
    """
    Concatenate feature vectors together in a way that they can be handed into
    a linear layer
    :param items A list of ag.Variables which are vectors
    :return One long row vector of all of the items concatenated together
    """
    return torch.cat(items, 1).view(1, -1)


def initialize_with_pretrained(pretrained_embeds, word_embedding_component):
    """
    Initialize the embedding lookup table of word_embedding_component with the embeddings
    from pretrained_embeds.
    Remember that word_embedding_component has a word_to_ix member you will have to use.
    For every word that we do not have a pretrained embedding for, keep the default initialization.
    :param pretrained_embeds dict mapping word to python list of floats (the embedding
        of that word)
    :param word_embedding_component The network component to initialize (i.e, a VanillaWordEmbeddingLookup
        or BiLSTMWordEmbeddingLookup)
    """
    # STUDENT
    for word, index in word_embedding_component.word_to_ix.items():
        if word in pretrained_embeds:
            word_embedding_component.word_embeddings.weight.data[index] = torch.Tensor(pretrained_embeds[word])
    # END STUDENT
