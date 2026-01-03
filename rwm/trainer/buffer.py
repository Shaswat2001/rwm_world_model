import numpy as np

class EpisodicReplayBuffer:

    def __init__(self, dim, buffer_size, device= "cpu"):
        
        self.dim = dim
        self.buffer_size = buffer_size
        self.device = device 

        self.buffer = None
        self.step = 0
        self.transitions = 0

    def _initilize(self):

        if isinstance(self.dim, list):
            self.buffer = [np.zeros((self.buffer_size, d)) if d > 0 else None for d in self.dim]
        else:
            assert self.dim > 0, "Dimension of the buffer should be positive"
            self.buffer = np.zeros((self.buffer_size, self.dim))

    def insert_buffer(self, buffer, input):

        num_inputs = input.shape[0]
        end_idx = self.step + num_inputs

        if end_idx > self.buffer_size:
            buffer[self.step:self.buffer_size] = input[:self.buffer_size-self.step]
            buffer[:end_idx-self.buffer_size] = input[self.buffer_size-self.step:]
        else:
            buffer[self.step: end_idx] = input

        return num_inputs

    def add(self, inputs):

        if self.buffer is None:
            self._initilize()

        num_inputs = 0

        if isinstance(self.buffer, list):
            for buffer, input in zip(self.buffer, inputs):
                if buffer is not None:
                    num_inputs = self.insert_buffer(buffer, input)
        else:
            num_inputs = self.insert_buffer(self.buffer, inputs)

        self.transitions = min(self.buffer_size, self.transitions + num_inputs)
        self.step = (self.step + num_inputs) % self.buffer_size

    def padding_buffer(self, sequence_length):

        padding_size = sequence_length - self.transitions

        if isinstance(self.buffer, list):
            return [np.concatenate([np.zeros((padding_size, buffer.shape[-1])), buffer[:self.transitions]], axis=1) if buffer is not None else None for buffer in self.buffer]
        else:
            return np.concatenate([np.zeros((padding_size, self.buffer.shape[-1])), self.buffer[:self.transitions]], axis=1)

    def where_terminate(self, buffer):

        indices = None
        done_buffer = buffer[-1] if isinstance(buffer, list) else None

        if done_buffer is not None:
            (indices, ) = np.nonzero(done_buffer > 0)

        return indices

    def sample(self, sequence_length, batch_size= 64):
        
        pad_buffer = self.padding_buffer(sequence_length)
        indices = self.where_terminate(pad_buffer)

        if indices is None:
            max_start_indices = max(self.transitions - sequence_length, 0) + 1
            sample_starts = np.random.choice(max_start_indices, size=batch_size)
            offsets = np.arange(sequence_length)
            if isinstance(pad_buffer, list):
                return [buf[sample_starts[:, None] + offsets] if buf is not None else None for buf in pad_buffer]
            else:
                return pad_buffer[sample_starts[:, None] + offsets]
            
        sampled_idxs = np.random.choice(len(indices), size=batch_size)
        sampled_starts = indices[sampled_idxs]
        offsets = np.arange(sequence_length)

        if isinstance(pad_buffer, list):
            return [buf[sampled_starts[:, None] + offsets] if buf is not None else None for buf in pad_buffer]
        else:
            return pad_buffer[sampled_starts[:, None] + offsets]

    

