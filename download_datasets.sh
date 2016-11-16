#!/bin/bash

BABI_TASKS=dataset/raw/babi_tasks_data_1_20_v1.2.tar.gz
DIALOG_TASKS=dataset/raw/dialog_babi_tasks_data_1_6.tgz
CHILDRENS_BOOK=dataset/raw/childrens_book_test.tgz
MOVIE_DIALOG=dataset/raw/movie_dialog_dataset.tgz
WIKIMOVIES=dataset/raw/wikimovies_dataset.tar.gz
DIALOG_LL=dataset/raw/dialog_based_LL_dataset.tgz
SIMPLE_QUESTIONS=dataset/raw/simple_questions_v2.tgz

wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz -O $BABI_TASKS
wget https://scontent-sjc2-1.xx.fbcdn.net/t39.2365-6/13437784_1766606076905967_221214138_n.tgz -O $DIALOG_TASKS
wget http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz -O $CHILDRENS_BOOK
wget http://www.thespermwhale.com/jaseweston/babi/movie_dialog_dataset.tgz -O $MOVIE_DIALOG
wget http://www.thespermwhale.com/jaseweston/babi/movieqa.tar.gz -O $WIKIMOVIES
wget http://www.thespermwhale.com/jaseweston/babi/dbll.tgz -O $DIALOG_LL
wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz -O $SIMPLE_QUESTIONS
