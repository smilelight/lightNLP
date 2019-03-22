import jieba


def handle_line(entity1, entity2, sentence, begin_e1_token='<e1>', end_e1_token='</e1>', begin_e2_token='<e2>',
                end_e2_token='</e2>'):
    assert entity1 in sentence
    assert entity2 in sentence
    sentence = sentence.replace(entity1, begin_e1_token + entity1 + end_e1_token)
    sentence = sentence.replace(entity2, begin_e2_token + entity2 + end_e2_token)
    sentence = ' '.join(jieba.cut(sentence))
    sentence = sentence.replace('< e1 >', begin_e1_token)
    sentence = sentence.replace('< / e1 >', end_e1_token)
    sentence = sentence.replace('< e2 >', begin_e2_token)
    sentence = sentence.replace('< / e2 >', end_e2_token)
    return sentence.split()


if __name__ == '__main__':
    test_str = '曾经沧海难为水哈哈谁说不是呢？！呵呵 低头不见抬头见'
    e1 = '沧海'
    e2 = '不是'
    print(list(jieba.cut(test_str)))
    print(handle_line(e1, e2, test_str))

