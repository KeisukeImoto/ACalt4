# AudioCaps Alternative Captions 

We created alternative captions for [AudioCaps](https://audiocaps.github.io/), AudioCaps Alternative 4 Captions (ACalt4).
While the files in this folder do not provide complete information about how we generate, they are for your reference for your future extended versions.

- [audiocaps_alternative_4.csv](audiocaps_alternative_4.csv)  The generated caption dataset.
- [gpt_example.pdf](gpt_example.pdf)          Example captions.
- [prompt.pdf](prompt.pdf)                    The prompt for ChatGPT for generation.
- [blip_to_generate.py](blip_to_generate.py)  The program excerpt used to generate image captions of each YouTube video using BLIP-2.
- [gpt_generation.py](gpt_generation.py)      The program excerpt used to generate alternative captions using the BLIP-2 captions and the AudioSet labels.

### References

- [AudioCaps] C. D. Kim, B. Kim, H. Lee, and G. Kim, “AudioCaps: Generating Captions for Audios in The Wild,” in NAACL-HLT, 2019.
- [BLIP-2] J. Li, D. Li, S. Savarese, and S. C. H. Hoi, “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models,” in ICML, 2023.
- [AudioSet] J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, “Audio Set: An ontology and human-labeled dataset for audio events,” in ICASSP, 2017, pp. 776–780.
