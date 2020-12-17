import game
import minimax
import graph
import random


def main():
    """Create the board and start the game."""
    ps = minimax.gen_states()
    pt = minimax.gen_table(ps)

    net = graph.FeedForwardNet(graph.activation_sigmoid, 0.01, 20, 10, 10, 10)

    for i in range(500):
        d = random.choice(pt)
        net.classify(d)
        net.update_weights(pt[d])

    for i in range(5):
        d = random.choice(pt)
        net.classify(d)

        print(f"Trying: {d}")
        print(f"Classified: {net.forward_ai[-1]}")
        print(f"Correct Classification: {pt[d]}")


if __name__ == "__main__":
    # execute only if run as a script
    main()
