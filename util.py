""" Utility functions """


def identity(x):
    return x


def nub(xs):
    """ Removes duplicates, maintaining original order. """
    return nub_by(xs, identity)


def nub_by(xs, key):
    """ Removes elements with duplicate keys, maintaining original order. """
    seen = set()

    def check_and_add(x):
        k = key(x)
        if k not in seen:
            seen.add(k)
            return True
        return False

    return (x for x in xs if check_and_add(x))


def chunked(xs, chunk_size):
    """ Splits a list into `chunk_size`-sized pieces. """
    xs = list(xs)
    return [
        xs[i:i + chunk_size]
        for i in range(0, len(xs), chunk_size)
    ]
