#!/usr/bin/env bash

set -eux

hatch run cov:erase

hatch run test:install
hatch run test-38:install

hatch run test:run-cov
hatch run test-38:run-cov

hatch run cov:combine
hatch run cov:report

