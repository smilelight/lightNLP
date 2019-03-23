def bis_cws(words, tags):
    assert len(words) == len(tags)
    poses = []

    for i, tag in enumerate(tags):
        if tag in ['B', 'S']:
            begin = i
        if i == len(tags) - 1:
            poses.append(''.join(words[begin: i + 1]))
        elif tags[i + 1] != 'I':
            poses.append(''.join(words[begin: i + 1]))
            begin = i + 1
    return poses
