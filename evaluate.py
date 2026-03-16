import csv
import time

import jiwer
import torch
import torchaudio
from wav2vec2decoder import Wav2Vec2Decoder


def evaluate(decoder, manifest_path, method):
    hypotheses = []
    references = []

    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio, sr = torchaudio.load(row["path"])
            assert sr == 16000
            hyp = decoder.decode(audio, method=method)
            hypotheses.append(hyp)
            references.append(row["text"])

    cer = jiwer.cer(references, hypotheses)
    wer = jiwer.wer(references, hypotheses)
    return cer, wer


def task1():
    print("Task 1")
    decoder = Wav2Vec2Decoder(lm_model_path=None)
    t0 = time.time()
    cer, wer = evaluate(decoder, "data/librispeech_test_other/manifest.csv", "greedy")
    print(f"  WER={wer:.2%}  CER={cer:.2%}  ({time.time()-t0:.1f}s)")


def task2():
    print("Task 2")
    for bw in [1, 3, 10, 50]:
        decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=bw)
        t0 = time.time()
        cer, wer = evaluate(
            decoder, "data/librispeech_test_other/manifest.csv", "beam"
        )
        elapsed = time.time() - t0
        print(f"  beam_width={bw:>2}  WER={wer:.2%}  CER={cer:.2%}  ({elapsed:.1f}s)")


def task3():
    print("Task 3")
    for temp in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        decoder = Wav2Vec2Decoder(lm_model_path=None, temperature=temp)
        t0 = time.time()
        cer, wer = evaluate(
            decoder, "data/librispeech_test_other/manifest.csv", "greedy"
        )
        print(f"  T={temp:.1f}  WER={wer:.2%}  CER={cer:.2%}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    import sys

    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3"]
    for t in tasks:
        {"1": task1, "2": task2, "3": task3}[t]()
        print()
