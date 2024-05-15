#!/usr/bin/env bash

set -eux

hatch run cov:erase
hatch run test:run-cov
hatch run cov:combine
hatch run cov:report

