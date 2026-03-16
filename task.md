Your task will be to implement the cryptographic steganography described in the attached paper ‘paper.pdf’.
You will write a single python script, main.py. Main will either “encode” or “decode” a steganographic message. 
If it is set to encode it will take a secret message+secret key to hide in a plain text prompt. 
If it is set to decode it will take the message and secret key and decode it.
You are provided with access to a models logprobs.  
You should manage packages with uv
You should use the environment provided to run the model (source ~/vllm-env/bin/activate)
You should only report back to me once you are certain the code is ready and tested, make sure you implement the scheme in the paper and that it is fully functioning before reporting back. 