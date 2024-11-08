# Run job in a remote TPU VM

# VM_NAME=kmh-tpuvm-v3-32-5
# # VM_NAME=kmh-tpuvm-v3-128-1
# ZONE=europe-west4-a

# VM_NAME=kmh-tpuvm-v4-8-7
# ZONE=us-central2-b

VM_NAME=kmh-tpuvm-v2-32-3
ZONE=us-central1-a

echo $VM_NAME $ZONE

CONFIG=tpu

# some of the often modified hyperparametes:
batch=2048
lr=0.00002
wd=0
ep=4000
warm=100
width=128
n_T=40
drop=0.15
optm=adam

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
TBNAME=${VM_NAME}_${CONFIG}_ep${ep}lr${lr}bsz${batch}_n${n_T}_dp${drop}_fm
JOBNAME=fm/${now}_${salt}_${TBNAME}

LOGDIR=/kmh-nfs-ssd-eu-mount/logs/$USER/$JOBNAME
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}

sudo chmod 777 /kmh-nfs-us-mount/data/cached  # for saving cached data

echo 'Log dir: '$LOGDIR
echo 'tb entry: '${TBNAME:10}:$LOGDIR  # remove the first 10 characters 'kmh-tpuvm-'

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
echo Current dir: $(pwd)

pip3 install tqdm

python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root=CIFAR \
    --config.model.image_size=32 \
    --config.model.out_channels=3 \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.lr_schedule=const \
    --config.weight_decay=${wd} \
    --config.optimizer=${optm} \
    --config.adam_b2=0.999 \
    --config.warmup_epochs=${warm} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=100 \
    --config.visualize_per_epoch=100 \
    --config.eval_per_epoch=100 \
    --config.fid.fid_per_epoch=200 \
    --config.model.base_width=${width} \
    --config.model.n_T=${n_T} \
    --config.fid.eval_only=False \
    --config.aug.use_edm_aug=True \
    --config.model.use_aug_label=True \
    --config.model.dropout=${drop} 
    --config.dataset.data_dir=/kmh-nfs-us-mount/data/cifar
" 2>&1 | tee -a $LOGDIR/output.log

