#!/bin/bash


cd ./synthetic/

python3 generate.py --seed 123


cd ../th1/

DATADIR="/scratches/sagarmatha_2/ktj21/data/crcns/th-1/data/"
SAVEDIR="/scratches/ramanujan_2/dl543/preprocessed/th1/"

python3 preprocess.py --datadir $DATADIR --savedir $SAVEDIR --session_id 140313 --mouse_id Mouse28

python3 select_units.py --datadir $SAVEDIR --session_id 140313 --mouse_id Mouse28 --phase wake



cd ../hc3/

DATADIR="/scratches/ramanujan_2/dl543/hc-3/"
SAVEDIR="/scratches/ramanujan_2/dl543/preprocessed/hc3/"

python3 preprocess.py --datadir $DATADIR --savedir $SAVEDIR --session_id ec014.468

python3 select_units.py --datadir $SAVEDIR --session_id ec014.468 --rec_id ec014.29
