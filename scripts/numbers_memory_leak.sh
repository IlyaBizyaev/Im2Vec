for _ in {1..22}; do
  CUDA_VISIBLE_DEVICES=0 python run.py -c configs/baseline_numbers_kl.yaml
done
