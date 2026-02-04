from transformers.cache_utils import DynamicCache
c = DynamicCache()
print('All methods:', [a for a in dir(c) if not a.startswith('_')])
