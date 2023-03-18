import random

def random_rank(lst):
    # lst is a list of items to be ranked
    # copy the list to avoid modifying the original
    lst_copy = lst[:]
    # create an empty list to store the rank
    rank = []
    # loop until the copy list is empty
    while lst_copy:
        # choose a random item from the copy list
        item = random.choice(lst_copy)
        # append it to the rank list
        rank.append(item)
        # remove it from the copy list
        lst_copy.remove(item)
    # return the rank list
    return rank