import collections

def init_graph():
    graph = [
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    return graph


def search(key, value, line):
    genres = ['jazz', 'rock', 'blues', 'metal', 'pop',
              'classical', 'reggae', 'disco', 'country',
              'alternative', 'folk', 'death metal',
              'folk rock', 'hip hop', 'rapcore', 'house']
    out_dict = collections.defaultdict(int)
    for i in xrange(0, len(line)):
        # check for each genre in line and add to dict
        if genres[i] != key and line[i] == 1:
            out_dict[genres[i]] = max(value - line[i], 0)
    # return 3 most common (key, value) items
    cnt = collections.Counter(out_dict)
    return cnt.most_common(len(out_dict.items()))


def recommend(in_dict, amount):
    # let's say we got input dict
    graph = init_graph()
    genres = ['jazz', 'rock', 'blues', 'metal', 'pop',
              'classical', 'reggae', 'disco', 'country',
              'alternative', 'folk', 'death metal', 'new metal',
              'folk rock', 'hip hop', 'rapcore', 'house']
    result_dict = []
    # we'll check for line in graph
    for item in in_dict.items():
        result_dict += search(item[0], item[1], graph[genres.index(item[0])])
    out_dict = collections.defaultdict(int)
    for item in result_dict:
        out_dict[item[0]] = max(item[1], out_dict[item[0]])
    return collections.Counter(out_dict).most_common(amount)