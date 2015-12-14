__author__ = 'Lucas Ramadan'

def dexplicit(l):

    # here we will store word conversions
    d = {'nigga': 'n****', 'shit': 's**t',
         'fuck':'f**k', 'ass': '@$$',
         'bitch': 'b***h', 'motherfuck': 'motherf***k',
         'motherfucker': 'motherf****r'}

    # go through each word
    for i, word in enumerate(l):
        # see if there is a replacement
        if word in d:
            # and replace it
            l[i] = d[word]

    return l
