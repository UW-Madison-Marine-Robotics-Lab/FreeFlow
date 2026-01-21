#!/usr/bin/env bash
set -euo pipefail

rsync -az --delete \
    --exclude .git/ \
    --exclude build*/ \
    --exclude .vscode/ \
    --exclude .DS_Store \
    --exclude docs/ \
    --exclude assets/configs/ \
    ./ labpc:~/Desktop/remote_projects/FreeFlow