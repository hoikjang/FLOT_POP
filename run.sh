#!/usr/bin/env bash

# ----------------------------------------------------------------------
# USE
# ----------------------------------------------------------------------
# bash bin/01_extract-holiday.sh /DATA/AI/SI/info/$(date '+%Y%m%d')/holiday_info.csv
# ----------------------------------------------------------------------

set -e


CURRENT_DATE_TIME=$(date '+%Y%m%d%H%M%S')
CURRENT_DATE=$(date '+%Y%m%d')
BASE_DIR=/home/ggits/mls/01.유동인구_밀집도_예측/ST_Resnet/Run
PYTHON_BIN=/home/ggits/anaconda3/envs/env1/bin/python
eval "$(conda shell.bash hook)"
conda activate env1


$PYTHON_BIN $BASE_DIR/etl_1.py --load_config $BASE_DIR/config.yaml --current_date $CURRENT_DATE --current_time $CURRENT_DATE_TIME --base_dir $BASE_DIR
$PYTHON_BIN $BASE_DIR/etl_2.py --load_config $BASE_DIR/config.yaml --current_date $CURRENT_DATE --current_time $CURRENT_DATE_TIME --base_dir $BASE_DIR
$PYTHON_BIN $BASE_DIR/modeling.py --load_config $BASE_DIR/config.yaml --current_date $CURRENT_DATE --current_time $CURRENT_DATE_TIME --base_dir $BASE_DIR
$PYTHON_BIN $BASE_DIR/prediction.py --load_config $BASE_DIR/config.yaml --current_date $CURRENT_DATE --current_time $CURRENT_DATE_TIME --base_dir $BASE_DIR


conda deactivate
