## 📁 core/ → Shared infrastructure
## 🎯 Purpose: Contains utilities and global configuration used everywhere.

Files inside:

```bash 
utils.py
```
contains basic low-level helpers:

i)   XOR

ii)  bit conversion

iii) rotations

```bash 
config.py
```

i)  Block size

ii) Delta (input difference)

iii) Default rounds
Sample counts

```bash 
dispatch.py
```
Maps cipher to calling function 

eg : "speck" → speck_encrypt