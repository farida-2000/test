import torch
from torchvggish_gpu import vggish
import vggish_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# For GPU support, the device must be cuda

embedding_model = vggish()
embedding_model.to(device)
embedding_model.eval()
example = vggish_input.wavfile_to_examples("bus_chatter.wav")
example = example.to(device)
audio_embeddings = embedding_model.forward(example)
print(audio_embeddings)
print(audio_embeddings.shape)
print(type(audio_embeddings))