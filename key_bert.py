#!/usr/bin/env python3

from my_transformers import KeyBERT, to_gpu, SentenceTransformer


DATUMS = [
    # "Hello, I'm a language model,",
    #load_asset("tokyo_completed")
    """Supervised learning is the machine learning task of learning a function that
maps an input to an output based on example input-output pairs. It infers a
function from labeled training data consisting of a set of training examples.
In supervised learning, each example is a pair consisting of an input object
(typically a vector) and a desired output value (also called the supervisory signal).
A supervised learning algorithm analyzes the training data and produces an inferred function,
which can be used for mapping new examples. An optimal scenario will allow for the
algorithm to correctly determine the class labels for unseen instances. This requires
the learning algorithm to generalize from the training data to unseen situations in a
'reasonable' way (see inductive bias)."""
]


def main() -> None:
    kw_model = KeyBERT(to_gpu(SentenceTransformer("all-MiniLM-L6-v2")))
    for DATA in DATUMS:
        keywords = kw_model.extract_keywords(DATA)
        print(keywords)


if __name__ == '__main__':
    main()
