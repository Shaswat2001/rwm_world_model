import numpy as np

class EpisodicReplayBuffer:

    def __init__(self, dim: dict, buffer_size: int, device: str= "cpu"):
        
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

        num_inputs = 1
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
            return [np.concatenate([np.zeros((padding_size, buffer.shape[-1])), buffer[:self.transitions]], axis=0) if buffer is not None else None for buffer in self.buffer]
        else:
            return np.concatenate([np.zeros((padding_size, self.buffer.shape[-1])), self.buffer[:self.transitions]], axis=0)

    def _generate_valid_indices(self, reset_buffer, sequence_length):

        T = len(reset_buffer)

        if T < sequence_length:
            return np.array([], dtype=np.int64)

        # Sliding window view: shape (T - seq_len + 1, seq_len)
        windows = np.lib.stride_tricks.sliding_window_view(
            reset_buffer.astype(bool), sequence_length, axis=0
        )

        # ignore reset at window start
        valid_mask = ~windows[:, 1:].any(axis=1)
        return np.nonzero(valid_mask)[0]

    def _ordered_buffer(self, buffer):
        if self.transitions < self.buffer_size:
            return buffer[:self.transitions]
        return np.concatenate(
            [buffer[self.step:], buffer[:self.step]],
            axis=0
        )

    def sample(self, sequence_length, batch_size= 64):

        assert isinstance(self.buffer, list), "Reset buffer must be last entry"
        assert self.buffer[-1].shape[-1] == 1 or self.buffer[-1].ndim == 1, \
            "Last buffer must be reset flags"

        if sequence_length > self.transitions:
            pad_buffer = self.padding_buffer(sequence_length)
        else:
            pad_buffer = self.buffer

        if isinstance(pad_buffer, list):
            pad_buffer = [
                self._ordered_buffer(buf) if buf is not None else None
                for buf in pad_buffer
            ]
        else:
            pad_buffer = self._ordered_buffer(pad_buffer)

        indices = self._generate_valid_indices(pad_buffer[-1], sequence_length)

        if len(indices) == 0:
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