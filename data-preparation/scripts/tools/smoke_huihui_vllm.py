import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
from vllm import LLM,SamplingParams

def main():
    p='/workspace/models/Huihui-Qwen3.8-27B-abliterated'
    m=LLM(model=p,dtype='bfloat16',tensor_parallel_size=2,gpu_memory_utilization=.6,max_model_len=8192,max_num_seqs=8,enforce_eager=True,gdn_prefill_backend='triton')
    t=m.get_tokenizer(); text=t.apply_chat_template([{'role':'user','content':'What is 2 + 2?'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    r=m.generate([text],SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)
    assert r and r[0].outputs and r[0].outputs[0].text.strip()
    print('SMOKE OK',flush=True)

if __name__ == "__main__":
    main()
