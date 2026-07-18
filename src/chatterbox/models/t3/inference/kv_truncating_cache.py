"""KV cache with max_position truncation for bucketed CUDA graph inference."""
from transformers import StaticCache


class LayerCacheProxy:
    """Proxy object to trick GPT2Model into using StaticCache's in-place update."""
    def __init__(self, cache, layer_idx):
        self.cache = cache
        self.layer_idx = layer_idx

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return self.cache.update(key_states, value_states, self.layer_idx, cache_kwargs)

    def __iter__(self):
        key, value = self.cache[self.layer_idx]
        yield key
        yield value

    def __getitem__(self, idx):
        return self.cache[self.layer_idx][idx]

    def __len__(self):
        return 2


class KVTruncatingStaticCache(StaticCache):
    """StaticCache that truncates the returned key/value states to ``_max_position``.

    During CUDA graph capture, ``T3HuggingfaceBackend.forward()`` sets
    ``cache._max_position = <bucket_size>`` before calling the model.  This
    ``update()`` override then slices the returned KV tensors so that each
    attention layer only attends to the *filled* portion of the cache (rounded
    up to the nearest bucket), rather than the full ``max_cache_len`` allocation.

    The truncation is a CUDA op and gets baked into the captured graph at
    capture time.  During replay the same fixed-size slice is executed, which
    is correct because each bucket has its own captured graph.
    """

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_states, value_states = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )
        max_pos = getattr(self, "_max_position", None)
        if max_pos is not None:
            key_states = key_states[:, :, :max_pos]
            value_states = value_states[:, :, :max_pos]
        return key_states, value_states

    def __getitem__(self, layer_idx: int):
        if hasattr(super(), "__getitem__"):
            key_states, value_states = super().__getitem__(layer_idx)
        else:
            key_states = self.key_cache[layer_idx]
            value_states = self.value_cache[layer_idx]
            
        max_pos = getattr(self, "_max_position", None)
        if max_pos is not None:
            key_states = key_states[:, :, :max_pos]
            value_states = value_states[:, :, :max_pos]
        return (key_states, value_states)

    def get_seq_length(self, layer_idx=0):
        max_pos = getattr(self, "_max_position", None)
        if max_pos is not None:
            return max_pos
        if hasattr(super(), "get_seq_length"):
            return super().get_seq_length(layer_idx)
        return self.max_cache_len

    def __iter__(self):
        for layer_idx in range(len(self.key_cache)):
            yield LayerCacheProxy(self, layer_idx)

    def get_max_length(self):
        max_pos = getattr(self, "_max_position", None)
        if max_pos is not None:
            return max_pos
        if hasattr(super(), "get_max_length"):
            return super().get_max_length()
        return self.max_cache_len
