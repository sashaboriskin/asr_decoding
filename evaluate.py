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


def task4():
    print("Task 4: Shallow fusion — alpha/beta sweep on LibriSpeech")
    decoder = Wav2Vec2Decoder(
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=10,
    )
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    print(f"  {'alpha':>6} {'beta':>5} {'WER':>8} {'CER':>8} {'time':>6}")
    for alpha in alphas:
        for beta in betas:
            decoder.alpha = alpha
            decoder.beta = beta
            t0 = time.time()
            cer, wer = evaluate(
                decoder, "data/librispeech_test_other/manifest.csv", "beam_lm"
            )
            elapsed = time.time() - t0
            print(f"  {alpha:>6.2f} {beta:>5.1f} {wer:>7.2%} {cer:>7.2%} {elapsed:>5.1f}s")


def task5():
    print("Task 5: 4-gram LM comparison on LibriSpeech")
    # Best alpha/beta from Task 4
    alpha, beta = 1.0, 1.0
    for lm_name, lm_path in [
        ("3-gram", "lm/3-gram.pruned.1e-7.arpa.gz"),
        ("4-gram", "lm/4-gram.arpa.gz"),
    ]:
        decoder = Wav2Vec2Decoder(
            lm_model_path=lm_path,
            beam_width=10,
            alpha=alpha,
            beta=beta,
        )
        t0 = time.time()
        cer, wer = evaluate(
            decoder, "data/librispeech_test_other/manifest.csv", "beam_lm"
        )
        print(f"  {lm_name}: WER={wer:.2%}  CER={cer:.2%}  ({time.time()-t0:.1f}s)")


def task6():
    print("Task 6: LM rescoring — alpha/beta sweep on LibriSpeech")
    decoder = Wav2Vec2Decoder(
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=10,
    )
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    print(f"  {'alpha':>6} {'beta':>5} {'WER':>8} {'CER':>8} {'time':>6}")
    for alpha in alphas:
        for beta in betas:
            decoder.alpha = alpha
            decoder.beta = beta
            t0 = time.time()
            cer, wer = evaluate(
                decoder, "data/librispeech_test_other/manifest.csv", "beam_lm_rescore"
            )
            elapsed = time.time() - t0
            print(f"  {alpha:>6.2f} {beta:>5.1f} {wer:>7.2%} {cer:>7.2%} {elapsed:>5.1f}s")


def task7():
    print("Task 7: Cross-domain comparison")
    # Best alpha/beta from Task 4 (SF)
    alpha, beta = 1.0, 1.0

    methods = [
        ("Greedy", "greedy", None),
        ("Beam search", "beam", None),
        ("Beam + 3-gram (SF)", "beam_lm", "lm/3-gram.pruned.1e-7.arpa.gz"),
        ("Beam + 3-gram (RS)", "beam_lm_rescore", "lm/3-gram.pruned.1e-7.arpa.gz"),
    ]
    datasets = [
        ("LibriSpeech", "data/librispeech_test_other/manifest.csv"),
        ("Earnings22", "data/earnings22_test/manifest.csv"),
    ]

    print(f"  {'Method':<25} {'LS WER':>8} {'LS CER':>8} {'E22 WER':>8} {'E22 CER':>8}")
    for name, method, lm_path in methods:
        decoder = Wav2Vec2Decoder(
            lm_model_path=lm_path,
            beam_width=10,
            alpha=alpha,
            beta=beta,
        )
        results = []
        for ds_name, ds_path in datasets:
            t0 = time.time()
            cer, wer = evaluate(decoder, ds_path, method)
            results.extend([wer, cer])
        print(
            f"  {name:<25} {results[0]:>7.2%} {results[1]:>7.2%}"
            f" {results[2]:>7.2%} {results[3]:>7.2%}"
        )


def task7b():
    print("Task 7b: Temperature sweep on Earnings22 (greedy vs beam+LM)")
    # Best alpha/beta from Task 4 (SF)
    alpha, beta = 1.0, 1.0
    temps = [0.5, 1.0, 1.5, 2.0]

    print(f"  {'T':>5} {'Greedy WER':>12} {'Beam+LM WER':>13}")
    for temp in temps:
        dec_greedy = Wav2Vec2Decoder(lm_model_path=None, temperature=temp)
        _, wer_g = evaluate(
            dec_greedy, "data/earnings22_test/manifest.csv", "greedy"
        )

        dec_lm = Wav2Vec2Decoder(
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=10,
            alpha=alpha,
            beta=beta,
            temperature=temp,
        )
        _, wer_lm = evaluate(
            dec_lm, "data/earnings22_test/manifest.csv", "beam_lm"
        )
        print(f"  {temp:>5.1f} {wer_g:>11.2%} {wer_lm:>12.2%}")


def task9():
    print("Task 9: Domain-specific LM comparison")
    # Best alpha/beta from Task 4 (SF)
    alpha, beta = 1.0, 1.0

    lms = [
        ("LibriSpeech 3-gram", "lm/3-gram.pruned.1e-7.arpa.gz"),
        ("Financial 3-gram", "lm/financial-3gram.arpa.gz"),
    ]
    datasets = [
        ("LibriSpeech", "data/librispeech_test_other/manifest.csv"),
        ("Earnings22", "data/earnings22_test/manifest.csv"),
    ]

    for lm_name, lm_path in lms:
        print(f"\n  LM: {lm_name}")
        decoder = Wav2Vec2Decoder(
            lm_model_path=lm_path,
            beam_width=10,
            alpha=alpha,
            beta=beta,
        )
        for ds_name, ds_path in datasets:
            cer_sf, wer_sf = evaluate(decoder, ds_path, "beam_lm")
            cer_rs, wer_rs = evaluate(decoder, ds_path, "beam_lm_rescore")
            print(
                f"    {ds_name:<15}"
                f" SF: WER={wer_sf:.2%} CER={cer_sf:.2%}"
                f"  RS: WER={wer_rs:.2%} CER={cer_rs:.2%}"
            )


if __name__ == "__main__":
    import sys

    task_map = {
        "1": task1, "2": task2, "3": task3,
        "4": task4, "5": task5, "6": task6,
        "7": task7, "7b": task7b, "9": task9,
    }

    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3"]
    for t in tasks:
        task_map[t]()
        print()
