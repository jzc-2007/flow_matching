# clean tmp
rm -rf tmp

# python3 script.py

PWD=$(pwd)
python3 main.py \
    --debug=False \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.dataset.root=CIFAR \
    --config.model.image_size=32 \
    --config.model.out_channels=3 \
    --config.batch_size=512 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.learning_rate=0.001 \
    --config.lr_schedule=const \
    --config.weight_decay=0 \
    --config.optimizer=adamw \
    --config.adam_b2=0.999 \
    --config.warmup_epochs=200 \
    --config.log_per_step=100 \
    --config.eval_per_epoch=1 \
    --config.visualize_per_epoch=2 \
    --config.num_epochs=4000 \
    --config.model.base_width=128 \
    --config.model.n_T=18 \
    --config.fid.on_use=True \
    --config.fid.eval_only=False \
    --config.fid.fid_per_epoch=250 \
    --config.fid.device_batch_size=128 \
    --config.model.dropout=0.13 \
    --config.aug.use_edm_aug=True \
    --config.model.use_aug_label=True \
    --config.dataset.data_dir=/kmh-nfs-ssd-eu-mount/code/zhichengjiang/data/cifar


    # --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/edm/20240821_061448_xnrl14_kmh-tpuvm-v4-8-7_tpu_b512_constlr0.001_wd0_adam_ep4000wm200_w128_n18_uncond_fid_dense100/checkpoint_58200
    # --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/edm/20240821_021434_1mcmpd_kmh-tpuvm-v4-8-7_tpu_b512_constlr0.001_wd0_adam_ep4000wm200_w128_n35_uncond_vis_fid/checkpoint_58200
    # --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/edm/20240820_042616_wjfafl_kmh-tpuvm-v3-32-5_tpu_b512_constlr0.001_wd0_adam_ep4000wm200_w128_n35_uncond_vis
    # --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/edm/20240821_011215_kz4y20_kmh-tpuvm-v3-128-1_tpu_b512_constlr0.001_wd0_adam_ep4000wm200_w128_n35_uncond_vis_fid/checkpoint_97000

    # --config.model.net_type='context' \
    # --config.model.net_type='ncsnpp' \
    # --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/edm/20240820_023513_643o22_kmh-tpuvm-v4-8-7_tpu_b512_constlr0.001_wd0_adam_ep4000wm200_w128_uncond





