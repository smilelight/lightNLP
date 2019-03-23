def bis_pos(words, tags):
    assert len(words) == len(tags)
    poses = []

    for i, tag in enumerate(tags):
        if tag.split('-')[0] in ['B', 'S']:
            begin = i
            temp_type = tag.split('-')[1]
        if i == len(tags) - 1:
            poses.append((''.join(words[begin: i + 1]), temp_type))
        elif tags[i + 1].split('-')[0] != 'I' or tags[i + 1].split('-')[1] != temp_type:
            poses.append((''.join(words[begin: i + 1]), temp_type))
            begin = i + 1
            temp_type = tags[i + 1].split('-')[1]
    return poses
