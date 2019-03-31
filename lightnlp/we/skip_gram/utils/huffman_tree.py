"""
    构建霍夫曼树
"""


class HuffmanNode:
    def __init__(self, word_id, frequency):
        self.word_id = word_id  # 叶子结点存词对应的id, 中间节点存中间节点id
        self.frequency = frequency  # 存单词频次
        self.left_child = None
        self.right_child = None
        self.father = None
        self.Huffman_code = []  # 霍夫曼码（左1右0）
        self.path = []  # 根到叶子节点的中间节点id


class HuffmanTree:
    def __init__(self, wordid_frequency_dict):
        self.word_count = len(wordid_frequency_dict)  # 单词数量
        self.wordid_code = dict()
        self.wordid_path = dict()
        self.root = None
        unmerge_node_list = [HuffmanNode(wordid, frequency) for wordid, frequency in
                             wordid_frequency_dict.items()]  # 未合并节点list
        self.huffman = [HuffmanNode(wordid, frequency) for wordid, frequency in
                        wordid_frequency_dict.items()]  # 存储所有的叶子节点和中间节点
        # 构建huffman tree
        self.build_tree(unmerge_node_list)
        # 生成huffman code
        self.generate_huffman_code_and_path()

    def merge_node(self, node1, node2):
        sum_frequency = node1.frequency + node2.frequency
        mid_node_id = len(self.huffman)  # 中间节点的value存中间节点id
        father_node = HuffmanNode(mid_node_id, sum_frequency)
        if node1.frequency >= node2.frequency:
            father_node.left_child = node1
            father_node.right_child = node2
        else:
            father_node.left_child = node2
            father_node.right_child = node1
        self.huffman.append(father_node)
        return father_node

    def build_tree(self, node_list):
        while len(node_list) > 1:
            i1 = 0  # 概率最小的节点
            i2 = 1  # 概率第二小的节点
            if node_list[i2].frequency < node_list[i1].frequency:
                [i1, i2] = [i2, i1]
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        [i1, i2] = [i2, i1]
            father_node = self.merge_node(node_list[i1], node_list[i2])  # 合并最小的两个节点
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0, father_node)  # 插入新节点
        self.root = node_list[0]

    def generate_huffman_code_and_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            # 顺着左子树走
            while node.left_child or node.right_child:
                code = node.Huffman_code
                path = node.path
                node.left_child.Huffman_code = code + [1]
                node.right_child.Huffman_code = code + [0]
                node.left_child.path = path + [node.word_id]
                node.right_child.path = path + [node.word_id]
                # 把没走过的右子树加入栈
                stack.append(node.right_child)
                node = node.left_child
            word_id = node.word_id
            word_code = node.Huffman_code
            word_path = node.path
            self.huffman[word_id].Huffman_code = word_code
            self.huffman[word_id].path = word_path
            # 把节点计算得到的霍夫曼码、路径  写入词典的数值中
            self.wordid_code[word_id] = word_code
            self.wordid_path[word_id] = word_path

    # 获取所有词的正向节点id和负向节点id数组
    def get_all_pos_and_neg_path(self):
        positive = []  # 所有词的正向路径数组
        negative = []  # 所有词的负向路径数组
        for word_id in range(self.word_count):
            pos_id = []  # 存放一个词 路径中的正向节点id
            neg_id = []  # 存放一个词 路径中的负向节点id
            for i, code in enumerate(self.huffman[word_id].Huffman_code):
                if code == 1:
                    pos_id.append(self.huffman[word_id].path[i])
                else:
                    neg_id.append(self.huffman[word_id].path[i])
            positive.append(pos_id)
            negative.append(neg_id)
        return positive, negative


def test():
    word_frequency = {0: 4, 1: 6, 2: 3, 3: 2, 4: 2}
    print(word_frequency)
    tree = HuffmanTree(word_frequency)
    print(tree.wordid_code)
    print(tree.wordid_path)
    for i in range(len(word_frequency)):
        print(tree.huffman[i].path)
    print(tree.get_all_pos_and_neg_path())


if __name__ == '__main__':
    test()
