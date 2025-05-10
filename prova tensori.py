import tntorch as tn

t = tn.ones(64, 64)  # 64 x 64 tensor, filled with ones
t = t[:, :, None] + 2*t[:, None, :]  # Singleton dimensions, broadcasting, and arithmetics
print(tn.mean(t))  # Result: 3
