#!/bin/bash
# 마지막 git pull(fetch/merge) 시점을 확인. 로컬/HPC 어디서든 저장소 루트에서 실행.
set -eu

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "git 저장소가 아닙니다: $(pwd)" >&2
    exit 1
fi

echo "=== 저장소: $(git rev-parse --show-toplevel) ==="
echo

if [ -f .git/FETCH_HEAD ]; then
    echo "--- 마지막 fetch (git fetch 또는 git pull의 fetch 단계) ---"
    if stat --version > /dev/null 2>&1; then
        stat -c '%y  %n' .git/FETCH_HEAD   # GNU stat (Linux/HPC)
    else
        stat -f '%Sm  %N' .git/FETCH_HEAD  # BSD/mac stat
    fi
    echo "  내용: $(cat .git/FETCH_HEAD | head -1)"
else
    echo "--- .git/FETCH_HEAD 없음 (아직 fetch/pull 이력 없음) ---"
fi
echo

echo "--- reflog상 마지막 pull(merge) ---"
last_pull=$(git reflog show HEAD | grep -i "pull" | head -1 || true)
if [ -n "$last_pull" ]; then
    commit_hash=$(echo "$last_pull" | cut -d' ' -f1)
    echo "  $last_pull"
    echo "  커밋 시각: $(git log -1 --format='%cd' --date=iso "$commit_hash" 2>/dev/null || echo '(확인 불가)')"
else
    echo "  reflog에 pull 기록 없음"
fi
echo

echo "--- 현재 브랜치 vs origin ---"
branch=$(git rev-parse --abbrev-ref HEAD)
echo "  현재 브랜치: $branch"
if git rev-parse --verify "origin/$branch" > /dev/null 2>&1; then
    ahead=$(git rev-list --count "origin/$branch..$branch")
    behind=$(git rev-list --count "$branch..origin/$branch")
    echo "  origin/$branch 대비: ahead=$ahead behind=$behind"
    echo "  (behind > 0 이면 origin에 아직 안 받은 새 커밋이 있다는 뜻 - git fetch로 갱신 후 확인)"
else
    echo "  origin/$branch 없음 (fetch 먼저 필요할 수 있음)"
fi
echo

echo "--- 마지막 커밋 ---"
git log -1 --format='  %H  %cd  %s' --date=iso
