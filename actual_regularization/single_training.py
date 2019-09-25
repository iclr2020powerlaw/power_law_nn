from training import ex
import random
import argparse
from random_words import RandomWords
import uuid
import os




def main():
    sacredObj = ex.run(config_updates={**args},
                   options={'--name': name})
    return sacredObj.result


if __name__ == '__main__':
    main()
