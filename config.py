from typing import Dict

# constants from paper
_LEAKY_RELU_ALPHA = 0.1
_IMG_DIM = 608


def parse_cfg(cfg_file: str) -> Dict[str, str]:
    """Parses the config file and returns a list of blocks, 
    where each block is represented as a dictionary.
    """

    file = open(cfg_file, 'r')
    lines = file.read().split('\n')
    lines = [x.rstrip().lstrip()
             for x in lines if (len(x) > 0 and x[0] != '#')]

    # maintain list of blocks
    blocks = []

    block = {}
    for line in lines:
        if line[0] == "[":
            # marks new block
            if len(block) != 0:
                # append old block
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks
