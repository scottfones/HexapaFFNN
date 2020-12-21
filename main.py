import game
import minimax
import graph
import random


def main():
    """Entry point for script.

    1. Demonstate policy table

    2. Defines networks for adder and hexapawn, then test.
    """
    print(f"POLICY TABLE (n=70):\n{minimax.create_policytable()}")

    print("\n\nADDER NETWORK")
    net = graph.FeedForwardNet(graph.activation_relu, 0.01, 1, 2, 2, 2)
    graph.test_adder(net)

    print("\n\nHEXAPAWN NETWORK")
    net = graph.FeedForwardNet(graph.activation_relu, 0.01, 20, 13, 10, 9)
    graph.test_hexa(net)


if __name__ == "__main__":
    # execute only if run as a script
    main()
