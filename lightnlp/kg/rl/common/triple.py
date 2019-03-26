class Triple(object):
    def __init__(self, head, rel, tail):
        self.head = head
        self.rel = rel
        self.tail = tail
    
    @staticmethod
    def cpm_list(a, b):
        assert type(b) is Triple
        return min(a.head, a.tail) > min(b.head, b.tail)
    
    @staticmethod
    def cmp_head(a, b):
        assert type(b) is Triple
        return a.head < b.head or \
            a.head == b.head and a.rel < b.rel or \
            a.head == b.head and a.rel == b.rel and a.tail < b.rel
    
    @staticmethod
    def cmp_tail(a, b):
        assert type(b) is Triple
        return a.tail < b.tail or \
            a.tail == b.tail and a.rel < b.rel or \
            a.tail == b.tail and a.rel == b.rel and a.head < b.head
    
    @staticmethod
    def cmp_rel(a, b):
        assert type(b) is Triple
        return a.head < b.head or \
            a.head == b.head and a.tail < b.tail or \
            a.head == b.head and a.tail == b.tail and a.rel < b.rel


if __name__ == '__main__':
    a = Triple(1, 2, 3)
    b = Triple(2, 3, 4)
    print(Triple.cpm_list(a, b))
    print(Triple.cmp_head(a, b))
    print(Triple.cmp_tail(a, b))
    print(Triple.cmp_rel(a, b))
