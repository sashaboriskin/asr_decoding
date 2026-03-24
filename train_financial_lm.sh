#!/bin/bash
set -e

KENLM_BUILD="/tmp/kenlm_build/build/bin"
CORPUS="data/earnings22_train/corpus.txt"
OUTPUT="lm/financial-3gram.arpa.gz"

# Build KenLM if needed
if [ ! -f "$KENLM_BUILD/lmplz" ]; then
    echo "Building KenLM tools..."
    git clone --depth=1 https://github.com/kpu/kenlm /tmp/kenlm_build
    mkdir -p /tmp/kenlm_build/build
    cd /tmp/kenlm_build/build
    cmake .. && make -j4 lmplz build_binary
    cd -
fi

echo "Training 3-gram financial LM from $CORPUS ..."
$KENLM_BUILD/lmplz -o 3 --discount_fallback < "$CORPUS" > /tmp/financial-3gram.arpa

echo "Compressing to $OUTPUT ..."
gzip -c /tmp/financial-3gram.arpa > "$OUTPUT"

echo "Done: $OUTPUT"
