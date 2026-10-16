#!/usr/bin/env bash
# Download the BRISC dataset (Piana et al. 2026, Zenodo 10.5281/zenodo.18714621, CC BY 4.0)
# into this folder and check the MD5 sums.  Each antenna*_pos*.mat is ~2.1 GB.
#
#   bash download_brisc.sh                 # log-periodic RX antenna, TX positions 1..9 (~19 GB)
#   bash download_brisc.sh 1 2 3           # only these positions (~6.4 GB)
#   ANT=Dipole bash download_brisc.sh 1    # dipole RX antenna instead
#   PAR=4 bash download_brisc.sh           # files downloaded in parallel (default 2)
#
# Zenodo throttles single connections (~0.5 MB/s) and drops long transfers
# ("curl: (18) end of response with ... bytes missing").  This script therefore
#   * downloads several files at once (PAR),
#   * uses aria2c with 16 connections per file if it is installed (much faster:
#     conda install -c conda-forge aria2   or   sudo yum/apt install aria2),
#   * otherwise resumes with curl -C - in a loop until the size matches,
#   * can simply be run again: complete files are skipped, partial ones resumed.
# Afterwards:  cd ../.. && python -m mpr.brisc prepare --data data/brisc --antenna log
cd "$(dirname "$0")"
REC=18714621
ANT="${ANT:-Log}"
PAR="${PAR:-2}"
POS=("$@"); [ ${#POS[@]} -eq 0 ] && POS=(1 2 3 4 5 6 7 8 9)
FILES=(configurations_10000.mat)
for p in "${POS[@]}"; do FILES+=("antenna${ANT}_pos${p}.mat"); done

curl -sSL --retry 5 "https://zenodo.org/api/records/${REC}" -o .zenodo_record.json || { echo "cannot reach zenodo"; exit 1; }
# expected sizes:  name<TAB>size
python3 -c '
import json
for f in json.load(open(".zenodo_record.json"))["files"]:
    print(f["key"], f["size"], sep="\t")' > .sizes.tsv

dl() {   # download one file until its size matches the Zenodo record
  f="$1"; url="https://zenodo.org/records/18714621/files/${f}?download=1"
  want=$(awk -F'\t' -v k="$f" '$1==k{print $2}' .sizes.tsv)
  [ -z "$want" ] && { echo "!! $f not in the Zenodo record"; return 1; }
  for try in $(seq 1 100); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && { echo "== $f complete ($want bytes)"; return 0; }
    if [ "$have" -gt "$want" ]; then echo "!! $f larger than expected, restarting"; rm -f "$f" "$f.aria2"; fi
    echo "== $f: attempt $try, have $have / $want bytes"
    if command -v aria2c >/dev/null; then
      # a partial file without aria2's control file (e.g. left by curl) cannot be
      # resumed with 16 connections -> start it again instead of letting aria2c
      # rename the new download to "<name>.1.mat"
      if [ "$have" -gt 0 ] && [ ! -f "$f.aria2" ]; then rm -f "$f"; fi
      aria2c -c -x 16 -s 16 --file-allocation=none --max-tries=20 --retry-wait=10 \
             --auto-file-renaming=false --allow-overwrite=true \
             --summary-interval=60 --console-log-level=warn -o "$f" "$url"
    else
      curl -sSL --fail -C - --retry 10 --retry-delay 10 --connect-timeout 30 \
           --speed-limit 10000 --speed-time 120 -o "$f" "$url"
    fi
    sleep 5
  done
  echo "!! $f: giving up after 100 attempts"; return 1
}
export -f dl

printf '%s\n' "${FILES[@]}" | xargs -P "$PAR" -I{} bash -c 'dl "$@" > ".log_$1" 2>&1; tail -1 ".log_$1"' _ {}

python3 - "${FILES[@]}" <<'PY'
import hashlib, json, sys
rec = json.load(open(".zenodo_record.json"))
want = {f["key"]: f["checksum"].split(":", 1)[1] for f in rec["files"]}
bad = 0
for name in sys.argv[1:]:
    try:
        h = hashlib.md5()
        with open(name, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        ok = h.hexdigest() == want.get(name)
    except FileNotFoundError:
        ok = False
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {name}")
if bad:
    print(f"{bad} file(s) incomplete or corrupt: run the script again (it resumes); "
          "a corrupt complete file must be deleted first.")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && echo "done. Next: (cd ../.. && python -m mpr.brisc prepare --data data/brisc --antenna ${ANT,,})"
