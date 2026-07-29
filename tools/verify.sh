#!/usr/bin/env bash
# Full verification sweep: every Python script's check ledger, then the ES modules.
set -u
cd "$(dirname "$0")/.."
total_p=0; total_t=0; bad=0
run () {
  line=$(uv run python "$1" 2>&1 | grep -E '^[0-9]+/[0-9]+ PASS$' | tail -1)
  if [ -z "$line" ]; then printf '  %-34s %s\n' "$(basename "$1")" "NO LEDGER / ERROR"; bad=1; return; fi
  p=${line%%/*}; t=${line#*/}; t=${t%% *}
  printf '  %-34s %s\n' "$(basename "$1")" "$line"
  total_p=$((total_p+p)); total_t=$((total_t+t))
  [ "$p" != "$t" ] && bad=1
}
echo "python check ledgers"
run python/e3.py
for f in python/experiments/*.py; do run "$f"; done
echo "  ----------------------------------------------------"
printf '  %-34s %d/%d PASS\n' "TOTAL" "$total_p" "$total_t"
echo
echo "es modules"
node tools/checkmods.mjs "$PWD" | sed 's/^/  /'
exit $bad
