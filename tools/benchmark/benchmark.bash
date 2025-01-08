MODEL="llama_moe_sbmm_forloop"
OUTPUT_FOLDER=".local/llama_moe_sbmm_forloop"
DEVICE="1xH100"
for REQUEST_RATE in 1 2 5 10 20 40 60 80 100
do
  python bench_perf.py --tokenizer meta-llama/Llama-3.1-8B --model $MODEL --request-rate $REQUEST_RATE --output $OUTPUT_FOLDER --metric-percentiles 25,50,90,99 --endpoint http://localhost:8080/ --num-prompts 300
  sleep 20
done
python output_to_hf.py --output-dir $OUTPUT_FOLDER --device $DEVICE