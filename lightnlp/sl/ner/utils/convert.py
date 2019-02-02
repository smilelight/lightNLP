def iob_ranges(words, tags):
    """
    IOB -> Ranges
    """
    assert len(words) == len(tags)
    ranges = []

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split('_')[0] == 'O':
            ranges.append({
                'entity': ''.join(words[begin: i + 1]),
                'type': temp_type,
                'start': begin,
                'end': i
            })

    for i, tag in enumerate(tags):
        if tag.split('_')[0] == 'O':
            pass
        elif tag.split('_')[0] == 'B':
            begin = i
            temp_type = tag.split('_')[1]
            check_if_closing_range()
        elif tag.split('_')[0] == 'I':
            check_if_closing_range()
    return ranges
