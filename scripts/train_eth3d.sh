export CUDA_VISIBLE_DEVICES=0
DATA_DIR="/mnt/data1/eth3d_processed/"
CHECKPOINT_DIR="/mnt/data1/regnerf_outputs/"

SCENES=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
for scene in "${SCENES[@]}" 
do
  mkdir -p ${CHECKPOINT_DIR}/
  rm -r "${CHECKPOINT_DIR}/${scene}/"
  python -m train \
    --gin_configs=configs/eth3d.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}/${scene}/'" \
    --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${scene}/'"
  python -m eval \
    --gin_configs=configs/eth3d.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}/${scene}/'" \
    --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${scene}/'"
done
