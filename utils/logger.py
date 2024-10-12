import logging

class ColoredFormatter(logging.Formatter):
    GREEN = "\033[92m"  # 緑色
    RED = "\033[91m"    # 赤色
    RESET = "\033[0m"   # リセット

    def format(self, record):
        levelname_color = self.GREEN if record.levelno == logging.INFO else self.RED
        record.levelname = f"{levelname_color}{record.levelname}{self.RESET}"
        return super().format(record)

class Logger():
    def __init__(self):
        self.logger = logging.getLogger()
        self.handler = logging.StreamHandler()
        self.formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

def main():
    logger = Logger().logger
    logger.info("This is an info message.")
    logger.error("This is an error message.")

if __name__ == "__main__":
    main()