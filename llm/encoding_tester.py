import tiktoken
import sys
sys.path.append('../')

import config

inp_default = "[[8, 1, 3], [8, 1, 3], [8, 1, 3]]"
inp_space =  "[[8 1 3] [8 1 3] [8 1 3]]"
inp_no_space =  "[[813] [813] [813]]"

inp_default = "[[0, 0, 0, 0], [0, 5, 0, 0]]"
inp_space =  "[[0 0 0 0] [0 5 0 0]]"
inp_comma =  "[[0,0,0,0],[0,5,0,0]]"
inp_no_space =  "[[0000][0500]]"
inp_char =  "[[a, a, a, a] [f, a, a, a]]"
inp_char2 =  "[[a,a,a,a][f,a,a,a]]"
inp_char3 =  "[[a a a a][f a a a]]"
inp_char4 =  "[[black,black,black,black][black,white,black,black]]"
test_inputs = [inp_default, inp_space, inp_comma, inp_char, inp_char2, inp_char3]

encoding = tiktoken.encoding_for_model(config.GPT_MODEL)
for inp in test_inputs:
    # Get tokens number due to rate limit 
    encoded = encoding.encode(inp)
    print(inp)
    print(encoded)
    print(len(encoded))
    print([encoding.decode_single_token_bytes(token) for token in encoded])
    print("\n")