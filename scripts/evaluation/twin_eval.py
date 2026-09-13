import sys

from text2motion.app.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
