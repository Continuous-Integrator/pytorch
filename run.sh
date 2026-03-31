set -ex
cd ~/local/torchtitan
rm -rf /tmp/torchinductor_aorenste
export TORCH_LOGS=output_code,+aot_graphs
export TORCH_DISTRIBUTED_COMPILE_ON_ONE_RANK=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
# export TORCH_LOGS=output_code,+aot_graphs,+torch.fx.experimental.proxy_tensor
# CODE_NAME=good.out
# if [ "${BAD:-0}" -ne "0" ]; then
#     CODE_NAME=bad.out
# fi
./run.sh 2>&1
# cat $(grep 'Output code written to' ~/local/torchtitan/logs/none_*/attempt_0/0/stderr.log | head -1 | sed -e 's/DEBUG: Output code written to: //') > ~/local/pytorch/$CODE_NAME
