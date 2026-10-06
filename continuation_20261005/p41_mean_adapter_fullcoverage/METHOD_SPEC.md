# P41 full-coverage causal temporal-mean anchored adapter

P41 repeats the P40 causal FIFO8 mean-anchored residual adapter with no architecture, loss, optimizer, or data change. It removes the short-schedule ambiguity by training exactly three full seeded permutations of all 18,412 eligible fit groups (55,236 updates), starting from the P39 fixed mean anchor. Only the final checkpoint is evaluated once on the frozen P4 fit15/cal5 retrieval contract.
