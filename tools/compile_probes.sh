#!/usr/bin/env bash
# Compile every XINT8 probe in a directory with ignite-xdna's graph compiler and print one verdict line each:
# the container's "Lowered ..." summary, or the compiler's final error line verbatim.
# Usage: tools/compile_probes.sh PROBE_DIR BUILD_DIR LOG_DIR [extra ignite-compile flags...]
# Runs from the ignite-xdna checkout (IGNITE_XDNA, default ../ignite-xdna) through its research-iron.sh.
set -u
probes=$(cd "$1" && pwd); build=$(mkdir -p "$2" && cd "$2" && pwd); logs=$(mkdir -p "$3" && cd "$3" && pwd)
shift 3
xdna=${IGNITE_XDNA:-$(cd "$(dirname "$0")/../../ignite-xdna" && pwd)}
cd "$xdna" || exit 1
for f in "$probes"/*_xint8.onnx; do
    n=$(basename "$f" _xint8.onnx)
    log="$logs/compile_$n.log"
    bash scripts/research-iron.sh -m ignite_xdna.compiler.cli --engine graph --input "$(cygpath -w "$f")" \
        --output "$(cygpath -w "$build/$n.ignite")" "$@" > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        printf '%-12s OK   %s\n' "$n" "$(grep -a 'Lowered' "$log" | head -1 | sed 's/^\[\*\] //')"
    else
        printf '%-12s FAIL %s\n' "$n" "$(grep -aE '^[A-Za-z_.]*(Error|Exception)' "$log" | tail -1 | cut -c1-230)"
    fi
done
