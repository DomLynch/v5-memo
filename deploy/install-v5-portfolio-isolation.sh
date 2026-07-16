#!/bin/sh
set -eu

unit_dir=${SYSTEMD_UNIT_DIR:-/etc/systemd/system}
deploy_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dropin=zzz-v5-portfolio-isolated-fullraw.conf
shared_dropin=90-researka-shared-fullraw.conf

for unit in \
    v5-memo-portfolio-prepare.service \
    v5-memo-portfolio-catchup.service \
    v5-memo-portfolio-publish.service
do
    rm -f "$unit_dir/$unit.d/$shared_dropin"
    install -D -m 0644 \
        "$deploy_dir/$dropin" \
        "$unit_dir/$unit.d/$dropin"
done

systemctl daemon-reload
