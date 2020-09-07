def count_words(s, n):
    a = list(set(s.split(' ')))
    c = list()
    for i in a:
        c.append(tuple((i, s.split(' ').count(i))))
    c.sort(key=lambda self: self[1])
    # c.sort(key=itemgetter(1) after importing itemgetter
    print(c[0:n])


if __name__ == '__main__':
    count_words('s as as df df df', 2)
