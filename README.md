# HexapaFFNN
HexapaFFNN provides solutions to hexapawn using Feed Forward Neural Networks

Written with python 3.8.6. Imports numpy and random.

Running main.py as a script `python main.py` will display the policy table created by minimax, and test the adder and hexapawn networks.

I couldn't find a "good" network architecture for either the adder or hexapawn. The most obvious guess is that the gradients aren't being calculated correctly. The network seems to converge around equivalent values regardless of input. Perhaps I just constructed a FeedForwardNet class that lacks in scholastic aptitude, but would excel in other things? Underwater basket weaving, maybe?