normalize() {
    grep -v 'INFO - .....step:' |
        grep -v 'NCCL version 2' |
        grep -v -P 'Sleeping \d+ seconds for other ranks to complete' |
        grep -v -P 'NCCL WARN Call to bind failed: Address already in use' |
        perl -p -e 's/\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d/-timestamp-/g' |
        perl -p -e 's/\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d/-timestamp-/g' |
        perl -p -e 's/(object|dict) at 0x[0-9a-fA-F]+/$1 at ptr/g'
}

for RANK in 0 1 2 3 4 5 6; do
    NEXT=$(expr $RANK + 1)
    echo "--------------------------------------------------------------------------------"
    echo "--- rank $RANK vs rank $NEXT:"
    for F in stdout.log stderr.log; do
        echo "$F:"
        diff -wbBdu <(
            cat ~/local/torchtitan/logs/none_*/attempt_0/$RANK/$F | normalize
        ) <(
            cat ~/local/torchtitan/logs/none_*/attempt_0/$NEXT/$F | normalize
        )
    done
    OUTPUT_CODE_A=$(grep 'DEBUG: Output code written to:' ~/local/torchtitan/logs/none_*/attempt_0/$RANK/stderr.log | head -1 | sed -e 's/DEBUG: Output code written to: //')
    OUTPUT_CODE_B=$(grep 'DEBUG: Output code written to:' ~/local/torchtitan/logs/none_*/attempt_0/$NEXT/stderr.log | head -1 | sed -e 's/DEBUG: Output code written to: //')
    echo "--- OUTPUT CODE 0"
    diff -wbBdu $OUTPUT_CODE_A $OUTPUT_CODE_B

    OUTPUT_CODE_A=$(grep 'DEBUG: Output code written to:' ~/local/torchtitan/logs/none_*/attempt_0/$RANK/stderr.log | tail -1 | sed -e 's/DEBUG: Output code written to: //')
    OUTPUT_CODE_B=$(grep 'DEBUG: Output code written to:' ~/local/torchtitan/logs/none_*/attempt_0/$NEXT/stderr.log | tail -1 | sed -e 's/DEBUG: Output code written to: //')
    echo "--- OUTPUT CODE 1"
    diff -wbBdu $OUTPUT_CODE_A $OUTPUT_CODE_B
done
