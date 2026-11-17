import numpy as np


class Tokeniser:
    def __init__(self,
                 CONTEXT_LENGTH: int,
                 text: str | None = None,
                 tokens: list[str] | None = None):
        if text:
            self._buildVocabWords(text)
        else:
            self._str_to_int = {token: i for i, token in enumerate(tokens)}
            self._int_to_str = {i: token for i, token in enumerate(tokens)}
        self.num_tokens = len(self._str_to_int)
        self.CONTEXT_LENGTH = CONTEXT_LENGTH

    def __call__(self, text: np.ndarray[str]) -> np.ndarray:
        array = np.zeros((text.shape[0], self.CONTEXT_LENGTH), dtype=int)
        for i in range(text.shape[0]):
            tokens = self._tokeniseString(text[i])
            tokens = tokens[-self.CONTEXT_LENGTH:]
            while len(tokens) < self.CONTEXT_LENGTH:
                tokens.insert(0, 0)
            array[i] = np.array(tokens)
        return array

    def _tokeniseString(self, text: str) -> list:
        for char in "`¬!\"£$%^&*()_+-={}[]:@~;'#<>?,./\\|":
            text = text.replace(char, ' ' + char + ' ')
        while "  " in text:
            text = text.replace("  ", ' ')
        text = text.strip()
        indices = [0]
        for block in text.split("\n\n"):
            for line in block.splitlines():
                for word in line.strip().split(' '):
                    try:
                        indices.append(self._str_to_int[word])
                    except Exception as e:
                        print(repr(word))
                        print(repr(line))
                        raise e
                indices.append(self._str_to_int['\n'])
            if block != '':
                indices.pop(-1)
            indices.append(1)
            indices.append(0)
        indices = indices[:-2]
        return indices

    def _buildVocabWords(self, text: str) -> np.ndarray:
        for char in "`¬!\"£$%^&*()_+-={}[]:@~;'#<>?,./\\|":
            text = text.replace(char, ' ')
        text = text.replace("\n\n", ' ').replace('\n', " \n ")
        while "  " in text:
            text = text.replace("  ", ' ')
        text = text.strip()
        tokens = ["[BOS]", "[EOS]"] + [
            c for c in sorted(list(set(text.split(' '))))
        ]
        for char in "`¬!\"£$%^&*()_+-={}[]:@~;'#<>?,./\\|":
            tokens.append(char)
        self._str_to_int = {token: i for i, token in enumerate(tokens)}
        self._int_to_str = {i: token for i, token in enumerate(tokens)}

    def _buildVoacbBPE(self, text: str):
        tokens = []
        char_to_int = ["[BOS]", "[EOS]"] + [
            c for c in sorted(list(set(text)))
        ]
        for char in text:
            tokens.append(char_to_int.index(char))
        print("Tokenised by char")
        i = 0
        newline = False
        for i in range(len(tokens)):
            if tokens[i] == char_to_int.index('\n'):
                if newline:
                    tokens[i-1] = 1
                    tokens[i] = 0
                    newline = False
                else:
                    newline = True
        tokens.insert(0, 0)
        tokens.append(1)
        print("[BOS] and [EOS] processed")
        while len(char_to_int) < vocab_size:
            frequencies = {}
            for i in range(len(tokens)-1):
                if (tokens[i], tokens[i+1]) in frequencies.keys():
                    frequencies[(tokens[i], tokens[i+1])] += 1
                else:
                    frequencies[(tokens[i], tokens[i+1])] = 1
            reverse_frequencies = {
                i: pair for i, pair in zip(frequencies.values(), frequencies.keys())
            }
            top = max(reverse_frequencies.keys())
            to_merge = reverse_frequencies[top]
            i = 0
            hit_first = False
            print(f"Merging {to_merge}, ({char_to_int[to_merge[0]]}, {char_to_int[to_merge[1]]}) with frequeincy {top}")
            char_to_int.append(char_to_int[to_merge[0]] + char_to_int[to_merge[1]])
            while i < len(tokens):
                if tokens[i] == to_merge[1] and hit_first:
                    hit_first = False
                    tokens[i-1] = len(char_to_int) - 1
                    tokens.pop(i)
                elif tokens[i] == to_merge[0]:
                    hit_first = True
                    i += 1
                else:
                    i += 1
                    hit_first = False
        print("BPE Completed")
        print(len(char_to_int))
        print(char_to_int)
        # print(tokens)
        # print([char_to_int[t] for t in tokens])
        return char_to_int

    def tokenToString(self, token: int) -> str:
        return self._int_to_str[token]

    def stringToToken(self, string: str) -> int:
        return self._str_to_int[string]
