from transformers.cache_utils import DynamicCache
c = DynamicCache()
attrs = [a for a in dir(c) if 'seen' in a.lower() or 'seq' in a.lower() or 'length' in a.lower()]
print('Available:', attrs)
try:
    print('get_seq_length:', c.get_seq_length())
except Exception as e:
    print('Error:', e)
