import random
from pathlib import Path
from typing import Iterator, Tuple

from lxml import etree
from nltk.corpus import wordnet as wn


def raganato_file_iter(
    data_fname: str,
    gold_keys_fname: str,
    classify_tokens: Tuple[str, str] = ("<classify>", "</classify>"),
    gloss_tokens: Tuple[str, str] = ("<g>", "</g>"),
) -> Iterator[Tuple[str, int]]:
    with Path(gold_keys_fname).open(mode="r", encoding="utf-8") as gold_fd:
        xml_fd = etree.iterparse(
            data_fname,
            tag="sentence",
            events=("end",),
        )
        for _, sentence in xml_fd:
            phrase = []
            instances = []
            for idx, word in enumerate(sentence):
                phrase.append(word.text)
                if word.tag == "instance":
                    _, *keys = gold_fd.readline().split()
                    # Possibly more than one key is associated with current word, I willingly only take the first one
                    # Uncomment the following to consider all possible keys
                    # for key in keys:
                    instances.append((idx, keys, word.attrib["pos"]))

            for idx, keys, _pos in instances:
                key = keys[0]
                lemma = wn.lemma_from_key(key)
                syn = lemma.synset()
                correct_gloss = syn.definition()
                syns = wn.synsets(lemma.name())
                syns.remove(syn)
                if syns:  # at least 2 glosses are available
                    phrase_tmp = phrase[:]
                    phrase_tmp.insert(idx + 1, classify_tokens[1])
                    phrase_tmp.insert(idx, classify_tokens[0])
                    # Pick a random incorrect glos
                    incorrect_gloss = random.choice(syns).definition()
                    # insert incorrect and correct glosses in the phrase in random order
                    correct_index = random.randint(0, 1)
                    phrase_tmp.append(gloss_tokens[0])
                    phrase_tmp.append(
                        incorrect_gloss if correct_index else correct_gloss,
                    )
                    phrase_tmp.append(gloss_tokens[1])
                    phrase_tmp.append(gloss_tokens[0])
                    phrase_tmp.append(
                        correct_gloss if correct_index else incorrect_gloss,
                    )
                    phrase_tmp.append(gloss_tokens[1])
                    # yield the sentence
                    yield " ".join(phrase_tmp), correct_index

            # These instructions are needed because even using the iterator would form a huge tree
            # which would be, once again, impossible, for most machines to store in memory
            # Also, it's safe to call clear() here because no descendant will be accessed
            sentence.clear()

            # eliminate now-empty references from the root node to <Title> as well
            while sentence.getprevious() is not None:
                del sentence.getparent()[0]

        del xml_fd
