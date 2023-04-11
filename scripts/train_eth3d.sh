export CUDA_VISIBLE_DEVICES=0
DATA_DIR="/home/jae/data/eth3d_processed/"
CHECKPOINT_DIR="/mnt/d/regnerf/eth3d/"

SCENES=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
SCENES=(courtyard)
for scene in "${SCENES[@]}" 
do
  mkdir -p ${CHECKPOINT_DIR}/${scene}
  rm "${CHECKPOINT_DIR}/${scene}/"*
  python -m train \
    --gin_configs=configs/eth3d.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}/${scene}'" \
    --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${scene}'"
done
