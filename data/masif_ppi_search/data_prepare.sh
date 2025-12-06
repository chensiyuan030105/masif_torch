#!/bin/bash
set -euo pipefail

# Usage:
#   ./run_all.sh lists/full_list.txt

LIST="${1:?Usage: $0 <path/to/full_list.txt>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while IFS= read -r line; do
  # skip empty lines
  [[ -z "$line" ]] && continue

  # first whitespace-separated field
  field1="${line%%[[:space:]]*}"

  # parse PDBID_CHAIN1_CHAIN2
  IFS="_" read -r PDBID CHAIN1 CHAIN2 <<< "$field1"

  "$SCRIPT_DIR/data_prepare_one.sh" "${PDBID}_${CHAIN1}_${CHAIN2}"
done < "$LIST"

