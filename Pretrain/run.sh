CONFIG=sst_10sweeps_VS0.5_WS16_ED8_epochs288
bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py 8 --resume-from ./work_dirs/sst_10sweeps_VS0.5_WS16_ED8_epochs288/epoch_276.pth --work-dir ./work_dirs/$CONFIG/ --cfg-options evaluation.metric=nuscenes
