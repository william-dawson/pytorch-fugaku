import ctypes
import json
import os
from pathlib import Path

assert not os.environ.get('LD_PRELOAD')
libc=ctypes.CDLL(None)
libc.posix_memalign.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t,ctypes.c_size_t]
libc.free.argtypes=[ctypes.c_void_p]
ptr=ctypes.c_void_p()
size=256*1024*1024
assert libc.posix_memalign(ctypes.byref(ptr),64,size)==0
data=(ctypes.c_ubyte*size).from_address(ptr.value)
for offset in range(0,size,65536): data[offset]=1
lines=Path('/proc/self/smaps').read_text().splitlines()
selected=[]
active=False
for line in lines:
    first=line.split()[0]
    if '-' in first and ':' not in first:
        lo,hi=[int(x,16) for x in first.split('-')]
        active=lo<=ptr.value<hi
    if active:selected.append(line)
print(json.dumps(dict(address=ptr.value,bytes=size,smaps=selected,
    maps=Path('/proc/self/maps').read_text(),preload=os.environ.get('LD_PRELOAD')),indent=2))
libc.free(ptr)
