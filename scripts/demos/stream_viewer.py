from text2motion.app.cli import main

if __name__ == "__main__":
    main(["studio", *__import__("sys").argv[1:]])
