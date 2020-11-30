class FileLogger:
    def __init__(self, file_path, config_str):
        self.file_path = file_path
        self.config_str = config_str
        if file_path is not None:
            self.file = open(file_path, 'w')
            self.file.write(config_str + '\n')
        else:
            self.file = None
        print(config_str)

    def info(self, some_str):
        if self.file is not None:
            self.file.write(some_str + '\n')
            self.file.flush()
        print(some_str)

    def __call__(self, *some_str_list):
        combined_str = ' '.join([str(item) for item in some_str_list])
        self.info(combined_str)


def get_file_logger(file_path, config_str=''):
    fl = FileLogger(file_path, config_str)
    return fl