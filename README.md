
We have used Video Swin Transformer By [Ze Liu](https://github.com/zeliu98/)\*, [Jia Ning](https://github.com/hust-nj)\*, [Yue Cao](http://yue-cao.me),  [Yixuan Wei](https://github.com/weiyx16), [Zheng Zhang](https://stupidzz.github.io/), [Stephen Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en) and [Han Hu](https://ancientmooner.github.io/) to extract the visual embedding of the videos in I-CONECT.

["Video Swin Transformer"](https://arxiv.org/abs/2106.13230). It is based on [mmaction2](https://github.com/open-mmlab/mmaction2).

The Wav2vec base model is used for audio/speech embedding. The speech was sampled at 16 kHz to be compatible with Wav2vec model.
The vocab size is 32. Diarization (categorizing speaker1 vs speaker2 was done on the speech based on the start and end time of each sentence given by transcript). After extracting the patient's speech, all parts of the conversation related to the patient were concatenated together and a new audio file for only the patient was generated which was used for the speech extraction being fed to the wav2vec. 

BERT base uncased is used for word embedding extraction. 
For each experiment, we run the respective example_name of experiment/theme which is set up on 5 times running to address the generalizability of the model with different fold configurations.
The data (image frames, speech, transcript) is not included on Git Hub due to the privacy of patients (I-CONECT is a private dataset by Harvard Medical School). 

Each experiment was run on a CUDA environment using GTX 1080 GPUs on a server.

Preprocessing was employed on speech for the purpose of labeling interviewer vs interviewee based on the start and end time of each speaker in sentences given by the transcripts (the ASR was designed by I-CONECT and transcripts are included in the dataset).
The sentences of patients were concatenated to generate a single text file for each patient in the respective theme. CLS and SEP tokens were added to keep track of separation and the start of the sentence. 

Once the results were generated on the testset, the metrics are being calculated. 
