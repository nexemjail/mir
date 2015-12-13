import collections
import math


def init_graph():
    graph = [
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0]
    ]
    return graph


def floyd(in_graph):
    INF = 1000000

    dist = in_graph
    for i in xrange(0, 8):
        for j in xrange(0, 8):
            if dist[i][j] == 0 and i != j:
                dist[i][j] = INF

    for k in xrange(0, 8):
        for i in xrange(0, 8):
            for j in xrange(0, 8):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def search(key, value, line):
    genres = ['jazz', 'rock', 'blues', 'metal',
              'pop', 'classical', 'disco', 'country']
    out_dict = collections.defaultdict(int)

    for i in xrange(0, len(line)):
        # check for each genre in line and add to dict
        if genres[i] != key and line[i] > 0:
            out_dict[genres[i]] = max(value - line[i], 0)
            if value - line[i] == 0:
                out_dict[genres[i]] += 1

    return out_dict


def recommend(in_dict, amount):
    print in_dict

    # let's say we got input dict
    graph = floyd(init_graph())

    genres = ['jazz', 'rock', 'blues', 'metal',
              'pop', 'classical', 'disco', 'country']

    result_dict = collections.defaultdict(int)

    # we'll check for line in graph
    for item in in_dict.items():
        temp_dict = search(item[0], item[1], graph[genres.index(item[0])])
        for inner_item in temp_dict.items():
            result_dict[inner_item[0]] = max(result_dict[inner_item[0]], inner_item[1])

    count = 0
    for item in result_dict.items():
        count += item[1]

    if amount > count:
        while amount > count:
            for item in result_dict.items():
                if amount > count and result_dict[item[0]] > 0:
                    result_dict[item[0]] += 1
                    count += 1

    if amount < count:
        while amount < count:
            for item in result_dict.items():
                if amount < count and result_dict[item[0]] > 0:
                    result_dict[item[0]] -= 1
                    count -= 1

    return result_dict