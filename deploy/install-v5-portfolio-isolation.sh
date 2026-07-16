#!/bin/sh
set -eu

unit_dir=${SYSTEMD_UNIT_DIR:-/etc/systemd/system}
deploy_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dropin=zzz-v5-portfolio-isolated-fullraw.conf
shared_env=v5-memo-portfolio-shared-fullraw.env

install -D -m 0644 \
    "$deploy_dir/$shared_env" \
    /etc/v5-memo/portfolio-shared-fullraw.env

for unit in \
    v5-memo-portfolio-prepare.service \
    v5-memo-portfolio-catchup.service \
    v5-memo-portfolio-publish.service
do
    install -D -m 0644 \
        "$deploy_dir/$unit" \
        "$unit_dir/$unit"
    install -D -m 0644 \
        "$deploy_dir/$dropin" \
        "$unit_dir/$unit.d/$dropin"
done

systemctl daemon-reload
