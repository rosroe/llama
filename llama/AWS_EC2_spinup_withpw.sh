conda init bash
source .bashrc

conda activate pytorch
pip install --upgrade pip

MNTDIR="/mnt/s3-AKIASO7VLHTWYKGVMAF4/"
sudo mkdir $MNTDIR

echo AKIASO7VLHTWYKGVMAF4:xxx > .passwd-s3fs
chmod 400 .passwd-s3fs
#
git clone https://rosroe@github.com/rosroe/llama.git
export PYTHONPATH=~/llama

sudo apt update
sudo apt install s3fs

######################
######################

sudo s3fs rosroe1 $MNTDIR -o passwd_file=./.passwd-s3fs -o allow_other

LLAMADIR="$MNTDIR/llama/data_tmp"
echo $LLAMADIR
sudo chmod -R 777 $LLAMADIR/7B
sudo chmod 777 $LLAMADIR/tokenizer*

cd ~/llama
pip install -r requirements.txt

git checkout feature/simple_prompt
torchrun --nproc_per_node 1 example.py $LLAMADIR/7B/ $LLAMADIR/tokenizer.model --max_batch_size 1 -- max_gen_len 16
