class BaseGenerator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError("Subclasses must implement __next__")