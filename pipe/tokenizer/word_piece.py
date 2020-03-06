import glob

from tokenizers import BertWordPieceTokenizer


class BertWordPiece:

    def __init__(self, clean_text: bool, strip_accents: bool, lowercase: bool):
        self.clean = clean_text
        self.strip = strip_accents
        self.lower = lowercase

        self.tokenizer = BertWordPieceTokenizer(
            clean_text=self.clean, 
            strip_accents=self.clean
            lowercase=self.lower, 
            handle_chinese_chars=True
        )

    def train(self, files, vocab_size, min_frequency, limit_alphabet):
        self.trainer = self.tokenizer.train(
            files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            limit_alphabet=limit_alphabet,
            wordpieces_prefix="##",
        )

    def save(self, path, filename):
        self.tokenizer.save(path, filename)
