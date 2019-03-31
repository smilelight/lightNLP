from lightnlp.utils.word_vector import WordVectors

vector_path = '/home/lightsmile/Projects/MyGithub/lightNLP/examples/cbow_saves/cbow_base.bin'
word_vectors = WordVectors(vector_path)
print(word_vectors.get_similar_words('少女', dis_type='cos'))
