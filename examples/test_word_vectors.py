from lightnlp.utils.word_vector import WordVectors

vector_path = 'E:/Projects/myProjects/lightNLP/examples/cbow_saves/cbow_base.bin'
word_vectors = WordVectors(vector_path)
print(word_vectors.get_similar_words('少女', dis_type='cos'))
