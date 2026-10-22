#!/usr/bin/env bash
# Publish ONLY R4 source needed to reconstruct its exact adapter path.
# The current working tree is neither switched nor reset.
set -Eeuo pipefail

ROOT="/home/bbx/user_lx/GraphLoRA_collab_v1"
BRANCH="fix/r5-r3-bootstrap-20261022"
REMOTE="https://github.com/NanDingEi/GraphLoRA-Research.git"
cd "$ROOT"

git rev-parse --is-inside-work-tree >/dev/null
test -f minigpt4/models/minigpt4rec_v2.py
test -f experiments/collab_injection_v1/r3_paper_adapters.py || {
  echo "[R4-SYNC-ERROR] Original r3_paper_adapters.py not found."
  exit 2
}
echo "[R4-SYNC] Verifying original R4 source exists..."
grep -Eq 'r3_adapter|CoRA|CFLoRA|cora|cflora' experiments/collab_injection_v1/r3_paper_adapters.py || {
  echo "[R4-SYNC-ERROR] R3 module has no expected adapter identifiers."
  exit 2
}

git fetch "$REMOTE" "refs/heads/$BRANCH"
temp_dir="$(mktemp -d /tmp/graphlora-r4-source-XXXXXXXX)"
worktree_added=0
cleanup() {
  if [ "$worktree_added" = 1 ]; then
    git worktree remove --force "$temp_dir" || true
  fi
  rm -rf "$temp_dir"
}
trap cleanup EXIT
git worktree add --detach "$temp_dir" FETCH_HEAD
worktree_added=1

copy_one() {
  local rel="$1"
  mkdir -p "$temp_dir/$(dirname "$rel")"
  cp -p -- "$rel" "$temp_dir/$rel"
  echo "[R4-SYNC-FILE] $rel"
}
copy_one minigpt4/models/minigpt4rec_v2.py
copy_one experiments/collab_injection_v1/r3_paper_adapters.py

# Locate R4 experiment entrypoints based on concrete R4 log markers.
# Only .py files from the project root / experiments subtree can be copied.
files_found=0
while IFS= read -r -d '' f; do
  rel="${f#./}"
  case "$rel" in
    r5_*) continue;;
  esac
  if grep -Eq '\[R4-FREEZE\]|\[R4-SOURCE-OK\]|R4_ISOLATION_20261022|\[R3-MODEL\]' "$f"; then
    copy_one "$rel"
    files_found=$((files_found+1))
  fi
done < <(find . -maxdepth 1 -type f -name '*.py' -print0;
         if [ -d experiments ]; then
           find experiments -maxdepth 4 -type f -name '*.py' -print0
         fi)
echo "[R4-SYNC] R4 entrypoint candidates copied: $files_found"
if [ "$files_found" -eq 0 ]; then
  echo "[R4-SYNC-WARNING] No R4 runner matched the log markers; original adapter module was still copied."
fi

git -C "$temp_dir" add -- minigpt4/models/minigpt4rec_v2.py experiments
while IFS= read -r -d '' f; do
  rel="${f#./}"
  if [ -f "$temp_dir/$rel" ]; then git -C "$temp_dir" add -- "$rel"; fi
done < <(find . -maxdepth 1 -type f -name '*.py' -print0)

if git -C "$temp_dir" diff --cached --quiet; then
  echo "[R4-SYNC] Source already identical to branch; nothing to commit."
else
  git -C "$temp_dir" -c user.name="GraphLoRA R4 Source Sync" -c user.email="graphlora-r4-sync@users.noreply.github.com" commit -m "chore(R4): sync local original R3 adapter and R4 model source"
  git -C "$temp_dir" push "$REMOTE" "HEAD:refs/heads/$BRANCH"
fi
echo "[R4-SYNC-DONE] https://github.com/NanDingEi/GraphLoRA-Research/tree/$BRANCH"
