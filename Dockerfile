FROM registry.gitlab.com/fomoro/base:latest

MAINTAINER Jim Fleming <jim@fomoro.com>

RUN mkdir -p /var/entity_networks/entity_networks/
COPY entity_networks/ /var/entity_networks/entity_networks/
WORKDIR /var/entity_networks/
