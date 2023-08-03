# jax_custom_ops_and_custom_partitioning
To run from this directory:
docker run -it --gpus all  -v $PWD:/dir ghcr.io/nvidia/jax-te:nightly-2023-08-01
cd /dir
TEST_CASE=0 python -m pdb test.py
TEST_CASE=1 python -m pdb test.py
