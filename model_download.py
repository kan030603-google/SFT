from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('unsloth/Qwen3-8B-unsloth-bnb-4bit',cache_dir='/root/autodl-tmp/pretrained')
# model_dir = snapshot_download('Qwen/Qwen3-8B',cache_dir='/root/autodl-tmp/pretrained')