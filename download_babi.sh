#!/bin/bash

if [ ! -d ./data ]; then
  mkdir -p ./data
fi

BABI_TASKS=data/babi_tasks_data_1_20_v1.2.tar.gz

if [ ! -f $BABI_TASKS ]; then
  wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz -O $BABI_TASKS
fi
