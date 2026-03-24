import csv
import json
import sys
import time

import kenlm
import jiwer
import torch
import torchaudio
from wav2vec2decoder import Wav2Vec2Decoder

LIBRISPEECH = "data/librispeech_test_other/manifest.csv"
EARNINGS22 = "data/earnings22_test/manifest.csv"


def load_dataset(manifest_path):
    paths, refs = [], []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            refs.append(row["text"])
    return paths, refs


def forward_pass(decoder, paths, temperature=1.0):
    logits_list = []
    for p in paths:
        audio, sr = torchaudio.load(p)
        assert sr == 16000
        inputs = decoder.processor(audio, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = decoder.model(inputs.input_values.squeeze(0)).logits[0]
        logits_list.append(logits / temperature)
    return logits_list


def decode_all(decoder, logits_list, method):
    hyps = []
    for logits in logits_list:
        if method == "greedy":
            hyps.append(decoder.greedy_decode(logits))
        elif method == "beam":
            hyps.append(decoder.beam_search_decode(logits))
        elif method == "beam_lm":
            hyps.append(decoder.beam_search_with_lm(logits))
        elif method == "beam_lm_rescore":
            beams = decoder.beam_search_decode(logits, return_beams=True)
            hyps.append(decoder.lm_rescore(beams))
    return hyps


def wer_cer(refs, hyps):
    return round(jiwer.wer(refs, hyps), 5), round(jiwer.cer(refs, hyps), 5)


def save(results, path="results.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [saved]", flush=True)


def main():
    results = {}
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]

    ls_paths, ls_refs = load_dataset(LIBRISPEECH)
    e22_paths, e22_refs = load_dataset(EARNINGS22)

    print("Loading model...", flush=True)
    decoder = Wav2Vec2Decoder(
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=3,  # start with bw=3 for sweeps
    )

    print("Forward pass (LibriSpeech)...", flush=True)
    t0 = time.time()
    ls_logits = forward_pass(decoder, ls_paths)
    print(f"  {time.time()-t0:.0f}s", flush=True)

    print("Forward pass (Earnings22)...", flush=True)
    t0 = time.time()
    e22_logits = forward_pass(decoder, e22_paths)
    print(f"  {time.time()-t0:.0f}s", flush=True)

    # ==================== Task 4: SF sweep (bw=3) ====================
    print("\n=== Task 4: Shallow fusion sweep (bw=3) ===", flush=True)
    task4 = []
    for alpha in alphas:
        for beta in betas:
            decoder.alpha = alpha
            decoder.beta = beta
            t0 = time.time()
            hyps = decode_all(decoder, ls_logits, "beam_lm")
            wer, cer = wer_cer(ls_refs, hyps)
            print(f"  a={alpha:<5} b={beta:<4} WER={wer:.2%} CER={cer:.2%} ({time.time()-t0:.0f}s)", flush=True)
            task4.append({"alpha": alpha, "beta": beta, "wer": wer, "cer": cer})
    results["task4"] = task4
    best_sf = min(task4, key=lambda x: x["wer"])
    print(f"  >>> Best: a={best_sf['alpha']}, b={best_sf['beta']}, WER={best_sf['wer']:.2%}", flush=True)
    save(results)

    # ==================== Task 5: 3g vs 4g (bw=3) ====================
    print("\n=== Task 5: 3-gram vs 4-gram ===", flush=True)
    decoder.alpha = best_sf["alpha"]
    decoder.beta = best_sf["beta"]
    task5 = []
    for lm_name, lm_path in [("3-gram", "lm/3-gram.pruned.1e-7.arpa.gz"),
                               ("4-gram", "lm/4-gram.arpa.gz")]:
        decoder.lm_model = kenlm.Model(lm_path)
        hyps = decode_all(decoder, ls_logits, "beam_lm")
        wer, cer = wer_cer(ls_refs, hyps)
        print(f"  {lm_name}: WER={wer:.2%} CER={cer:.2%}", flush=True)
        task5.append({"lm": lm_name, "wer": wer, "cer": cer})
    results["task5"] = task5
    decoder.lm_model = kenlm.Model("lm/3-gram.pruned.1e-7.arpa.gz")
    save(results)

    # ==================== Task 6: Rescore sweep (bw=3) ====================
    print("\n=== Task 6: Rescoring sweep (bw=3) ===", flush=True)
    print("  Precomputing beams...", flush=True)
    t0 = time.time()
    all_beams_ls = [decoder.beam_search_decode(l, return_beams=True) for l in ls_logits]
    print(f"  {time.time()-t0:.0f}s", flush=True)

    task6 = []
    for alpha in alphas:
        for beta in betas:
            decoder.alpha = alpha
            decoder.beta = beta
            hyps = [decoder.lm_rescore(b) for b in all_beams_ls]
            wer, cer = wer_cer(ls_refs, hyps)
            print(f"  a={alpha:<5} b={beta:<4} WER={wer:.2%} CER={cer:.2%}", flush=True)
            task6.append({"alpha": alpha, "beta": beta, "wer": wer, "cer": cer})
    results["task6"] = task6
    best_rs = min(task6, key=lambda x: x["wer"])
    print(f"  >>> Best: a={best_rs['alpha']}, b={best_rs['beta']}, WER={best_rs['wer']:.2%}", flush=True)
    save(results)

    # ==================== Task 6 qualitative ====================
    print("\n=== Task 6: Qualitative (bw=10 for better hypotheses) ===", flush=True)
    decoder.beam_width = 10

    beam_hyps = decode_all(decoder, ls_logits, "beam")

    decoder.alpha = best_sf["alpha"]
    decoder.beta = best_sf["beta"]
    sf_hyps = decode_all(decoder, ls_logits, "beam_lm")

    decoder.alpha = best_rs["alpha"]
    decoder.beta = best_rs["beta"]
    all_beams_ls_10 = [decoder.beam_search_decode(l, return_beams=True) for l in ls_logits]
    rs_hyps = [decoder.lm_rescore(b) for b in all_beams_ls_10]

    qualitative = []
    for i in range(len(ls_refs)):
        if beam_hyps[i] != sf_hyps[i] or beam_hyps[i] != rs_hyps[i]:
            qualitative.append({
                "idx": i, "ref": ls_refs[i],
                "beam": beam_hyps[i], "sf": sf_hyps[i], "rs": rs_hyps[i],
            })
    results["task6_qualitative"] = qualitative[:15]
    print(f"  {len(qualitative)} samples differ", flush=True)
    for q in qualitative[:10]:
        print(f"  [{q['idx']}] REF:  {q['ref']}", flush=True)
        print(f"       BEAM: {q['beam']}", flush=True)
        if q["beam"] != q["sf"]:
            print(f"       SF:   {q['sf']}", flush=True)
        if q["beam"] != q["rs"]:
            print(f"       RS:   {q['rs']}", flush=True)
    save(results)

    # ==================== Task 7: Cross-domain (bw=10) ====================
    print("\n=== Task 7: Cross-domain comparison (bw=10) ===", flush=True)
    task7 = {}
    for ds_name, logits_list, refs in [
        ("librispeech", ls_logits, ls_refs),
        ("earnings22", e22_logits, e22_refs),
    ]:
        hyps = decode_all(decoder, logits_list, "greedy")
        wer, cer = wer_cer(refs, hyps)
        task7[f"greedy_{ds_name}"] = {"wer": wer, "cer": cer}
        print(f"  Greedy/{ds_name}: WER={wer:.2%} CER={cer:.2%}", flush=True)

        hyps = decode_all(decoder, logits_list, "beam")
        wer, cer = wer_cer(refs, hyps)
        task7[f"beam_{ds_name}"] = {"wer": wer, "cer": cer}
        print(f"  Beam/{ds_name}:   WER={wer:.2%} CER={cer:.2%}", flush=True)

        decoder.alpha = best_sf["alpha"]
        decoder.beta = best_sf["beta"]
        hyps = decode_all(decoder, logits_list, "beam_lm")
        wer, cer = wer_cer(refs, hyps)
        task7[f"sf_{ds_name}"] = {"wer": wer, "cer": cer}
        print(f"  SF/{ds_name}:     WER={wer:.2%} CER={cer:.2%}", flush=True)

        decoder.alpha = best_rs["alpha"]
        decoder.beta = best_rs["beta"]
        beams_list = [decoder.beam_search_decode(l, return_beams=True) for l in logits_list]
        hyps = [decoder.lm_rescore(b) for b in beams_list]
        wer, cer = wer_cer(refs, hyps)
        task7[f"rs_{ds_name}"] = {"wer": wer, "cer": cer}
        print(f"  RS/{ds_name}:     WER={wer:.2%} CER={cer:.2%}", flush=True)

    results["task7"] = task7
    save(results)

    # ==================== Task 7b: Temp sweep Earnings22 (bw=10) ====================
    print("\n=== Task 7b: Temperature sweep on Earnings22 ===", flush=True)
    task7b = []
    for temp in [0.5, 1.0, 1.5, 2.0]:
        e22_t = forward_pass(decoder, e22_paths, temperature=temp)

        hyps = decode_all(decoder, e22_t, "greedy")
        wer_g, cer_g = wer_cer(e22_refs, hyps)

        decoder.alpha = best_sf["alpha"]
        decoder.beta = best_sf["beta"]
        hyps = decode_all(decoder, e22_t, "beam_lm")
        wer_lm, cer_lm = wer_cer(e22_refs, hyps)

        print(f"  T={temp}: Greedy={wer_g:.2%}, Beam+LM={wer_lm:.2%}", flush=True)
        task7b.append({"T": temp, "greedy_wer": wer_g, "greedy_cer": cer_g,
                        "beam_lm_wer": wer_lm, "beam_lm_cer": cer_lm})
    results["task7b"] = task7b
    save(results)

    # ==================== Task 9: Domain LM comparison (bw=10) ====================
    print("\n=== Task 9: Domain LM comparison ===", flush=True)
    task9 = {}
    for lm_name, lm_path in [
        ("librispeech_3gram", "lm/3-gram.pruned.1e-7.arpa.gz"),
        ("financial_3gram", "lm/financial-3gram.arpa.gz"),
    ]:
        try:
            decoder.lm_model = kenlm.Model(lm_path)
        except Exception as e:
            print(f"  Skip {lm_name}: {e}", flush=True)
            continue

        for ds_name, logits_list, refs in [
            ("librispeech", ls_logits, ls_refs),
            ("earnings22", e22_logits, e22_refs),
        ]:
            decoder.alpha = best_sf["alpha"]
            decoder.beta = best_sf["beta"]
            hyps = decode_all(decoder, logits_list, "beam_lm")
            wer_sf, cer_sf = wer_cer(refs, hyps)

            decoder.alpha = best_rs["alpha"]
            decoder.beta = best_rs["beta"]
            beams = [decoder.beam_search_decode(l, return_beams=True) for l in logits_list]
            hyps = [decoder.lm_rescore(b) for b in beams]
            wer_rs, cer_rs = wer_cer(refs, hyps)

            task9[f"{lm_name}_{ds_name}"] = {
                "sf_wer": wer_sf, "sf_cer": cer_sf,
                "rs_wer": wer_rs, "rs_cer": cer_rs,
            }
            print(f"  {lm_name}/{ds_name}: SF={wer_sf:.2%} RS={wer_rs:.2%}", flush=True)

    results["task9"] = task9
    save(results)

    print(f"\n=== DONE ===", flush=True)
    print(f"Best SF: alpha={best_sf['alpha']}, beta={best_sf['beta']}", flush=True)
    print(f"Best RS: alpha={best_rs['alpha']}, beta={best_rs['beta']}", flush=True)


if __name__ == "__main__":
    main()
