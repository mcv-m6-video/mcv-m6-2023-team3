

def load_bb(path):
    """
    Loads the bounding boxes from a path.
    They list the ground truths of MTMC tracking in the MOTChallenge format 
    [frame, ID, left, top, width, height, 1, -1, -1, -1].
    Only left, top, width and height are used.
    """
    bbs = []
    with open(path, 'r') as f:
        for line in f:
            # Split the line by commas
            bb = line.strip().split(',')
            # Extract values 3, 4, 5, and 6
            values = [int(bb[i]) for i in range(2, 6)]
            bbs.append(values)
    return bbs