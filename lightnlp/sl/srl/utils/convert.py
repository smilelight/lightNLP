def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'rel':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def iob_ranges(words, tags):
    """
    IOB -> Ranges
    """
    assert len(words) == len(tags)
    events = {}

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split('-')[0] == 'O' or tags[i+1] == 'rel':
            events[temp_type] = ''.join(words[begin: i + 1])

    for i, tag in enumerate(tags):
        if tag == 'rel':
            events['rel'] = words[i]
        elif tag.split('-')[0] == 'O':
            pass
        elif tag.split('-')[0] == 'B':
            begin = i
            temp_type = tag.split('-')[1]
            check_if_closing_range()
        elif tag.split('-')[0] == 'I':
            check_if_closing_range()
    return events


def iobes_ranges(words, tags):
    new_tags = iobes_iob(tags)
    return iob_ranges(words, new_tags)
