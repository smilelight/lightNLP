import torch
from torchtext.vocab import Vectors

from .score_func import l1_score, l2_score, cos_score


class WordVectors(Vectors):

    def __init__(self, *args, **kwargs):
        super(WordVectors, self).__init__(*args, **kwargs)

    def get_similar_words(self, word: str, topk=3, dis_type='cos'):
        word_vector = self[word].view(1, -1)
        if dis_type == 'l1':
            dis_func = l1_score
            similar_vectors = torch.tensor([torch.exp(-1 * dis_func(word_vector, self[x].view(1, -1))) for x in self.stoi])
        elif dis_type == 'l2':
            dis_func = l2_score
            similar_vectors = torch.tensor([torch.exp(-1 * dis_func(word_vector, self[x].view(1, -1))) for x in self.stoi])
        else:
            dis_func = cos_score
            similar_vectors = torch.tensor([dis_func(word_vector, self[x].view(1, -1)) for x in self.stoi])

        topk_score, topk_index = torch.topk(similar_vectors, topk)
        topk_score = topk_score.cpu().tolist()
        topk_index = [self.itos[x] for x in topk_index]
        return list(zip(topk_index, topk_score))


if __name__ == '__main__':
    vector_path = '/home/lightsmile/Projects/MyGithub/lightNLP/examples/cbow_saves/cbow_base.bin'
    word_vectors = WordVectors(vector_path)
    print(word_vectors.get_similar_words('少女', dis_type='cos'))
