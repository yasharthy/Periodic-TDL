from collections import Counter

def _count_ctx(graphs):
    atom, bond = Counter(), Counter()
    N = len(graphs)
    for i, g in enumerate(graphs):
        # atoms
        toks = getattr(g["node_k1"], "ctx_label", None)
        assert toks != None, "ERROR! No atom context labels present!"
        atom.update([t for t in toks])
        # bonds (directed edges)
        rel = ("node_k1","via_edge","node_k1")
        toks = getattr(g[rel], "ctx_label", None)
        assert toks != None, "ERROR! No bond context labels present!"
        bond.update([t for t in toks])

        if i % 5000 == 0:
            print(f"Processed {i}/{N} molecules...")
        i+=1
                  
    return atom, bond

def _make_vocab(counter: Counter, min_freq=1, add_unk=True):
    vocab = {"__ignore__": 0}
    idx = 1
    if add_unk:
        vocab["<UNK>"] = idx
        idx += 1
    for tok, c in counter.most_common():
        if c < min_freq: break
        if tok in vocab:  continue
        vocab[tok] = idx
        idx += 1
    return vocab

def atom_and_bond_vocab(graphs, min_freq, add_unk=True):
    atom_c, bond_c = _count_ctx(graphs)
    atom_vocab = _make_vocab(atom_c, min_freq, add_unk)
    bond_vocab = _make_vocab(bond_c, min_freq, add_unk)
    return atom_vocab, bond_vocab
