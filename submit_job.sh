#!/usr/bin/env bash
gradient jobs create \
--name "$1" \
--container ${CONTAINER_NAME_BAYES} \
--registryUsername ${DOCKERHUB_USERNAME} \
--registryPassword ${DOCKERHUB_PASSWORD} \
--machineType "P6000" \
--command "./resolve.sh"