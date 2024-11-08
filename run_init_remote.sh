# initialize and set up remote TPU VM

VM_NAME=kmh-tpuvm-v3-32-3
ZONE=europe-west4-a  # v3

# VM_NAME=kmh-tpuvm-v2-32-2
# ZONE=us-central1-a  # v2

# install packages
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "

pip3 install absl-py==1.4.0
pip3 install clu==0.0.11
pip3 install flax==0.8.1
pip3 install jax[tpu]==0.4.25 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install ml-collections==0.1.1
pip3 install numpy==1.26.4
pip3 install optax==0.2.1
pip3 install tensorflow==2.15.0.post1
pip3 install torch==2.2.2
pip3 install torchvision==0.17.2
pip3 install orbax-checkpoint==0.4.4
pip3 install chex==0.1.86
pip3 install tqdm

"

# sanity check
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
python3 -c 'import jax; print(jax.device_count())'
"

# mount NFS Filestore
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "

sudo apt-get -y update
sudo apt-get -y install nfs-common

sudo mkdir -p /kmh-nfs-us-mount
sudo mount -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
sudo chmod go+rw /kmh-nfs-us-mount
ls /kmh-nfs-us-mount

sudo mkdir -p /kmh-nfs-ssd-eu-mount
sudo mount -o vers=3 10.150.179.250:/kmh_nfs_ssd_eu /kmh-nfs-ssd-eu-mount
sudo chmod go+rw /kmh-nfs-ssd-eu-mount
ls /kmh-nfs-ssd-eu-mount

"