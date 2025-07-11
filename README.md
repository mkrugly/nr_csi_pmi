# nr_csi_pmi

Ever struggled to understand how precoding matrices are generated from the 5G New Radio (NR) CSI Codebook Type II and Type II-Rel16 reports?
Or perhaps you've wondered how to generate PMI coefficients for beams selected from a DFT beam grid — or how different coefficients influence the resulting precoding matrix?

The `nr_csi_pmi` Python module is here to help.

It allows you to:
- Input PMI coefficients reported by the UE in CSI Codebook Type II or Type II-Rel16 reports,
or
- Manually specify DFT beam parameters.
  
The module then generates the corresponding precoding matrices according to 3GPP TS 38.214, specifically:
- Section 5.2.2.2.3 — Precoding for Type II codebooks with 2D antenna arrays
- Section 5.2.2.2.5 — Extensions for Rel-16
  
You’ll also see all intermediate computation steps, making it a valuable tool for learning, debugging, or prototyping.


## Examples

- Generation of CSI codebook typeII-rel16 PMI coefficients and corresponding precoding matrix for given DFT beam grid parameters:

```
    >>> import logging
    >>> logging.basicConfig(level=logging.INFO)
    >>> from nr_csi_pmi import rel16
    >>> gen_r16 = rel16.PmiGenerator(comb_inx=5, R=1, N3=18, v=1, N1N2=(4,3))
    >>> params = {
    ... "q1q2": (0, 1),
    ... "n1n2": ([2, 0, 1, 2], [1, 2, 2, 2]),
    ... "n3l": [[0, 1, 2, 3, 5]],
    ... "lifs_strongest": [(1, 1, 0)],
    ... "lifs_other": [(1, 0, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4)],
    ... "k1": [13],
    ... "k2": [5, 1, 3, 4],
    ... "c": [11, 10, 15, 12]
    ... }
    >>> pmi = gen_r16.beam_factory(**params)
    >>> pmi.log_summary()
    INFO:nr_csi_pmi.rel16:Coefficients: {'i11': 1, 'i12': 8, 'i23': [13], 'i18': [1], 'i15': 0, 'i16': [2378], 'i24': ['0xa5c0', None, None, None], 'i25': ['0xbafc', None, None, None], 'i17': ['0x4080201008', '', '', '']}
    INFO:nr_csi_pmi.rel16:Beam settings: {'q1q2': (0, 1), 'n1n2': ([2, 0, 1, 2], [1, 2, 2, 2]), 'n3l': [[0, 1, 2, 3, 5]], 'k1': [13], 'k2': [5, 1, 3, 4], 'c': [11, 10, 15, 12], 'lifs_strongest': [(1, 1, 0)], 'lifs_strongest_xpol': [(1, 5, 0)], 'lifs_other': [(1, 0, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4)]}
    INFO:nr_csi_pmi.rel16:Coefficients bin: {'i11': '0001', 'i12': '000001000', 'i23': '1101', 'i18': '001', 'i15': '0', 'i16': '100101001010', 'i24': '101001011100', 'i25': '1011101011111100', 'i17': '0100000010000000001000000001000000001000'}
    INFO:nr_csi_pmi.rel16:Coefficients str: [i11 0001, i12 000001000, i18 001, i23 1101, i15 0, i16 100101001010, i24 101001011100, i25 1011101011111100, i17 4080201008]
    >>> pmi.w.for_sb(0)
    array([[ 0.23281985-0.15816084j],
           [ 0.05206891-0.20519797j],
           [-0.40106993+0.02912044j],
           (...)
```

- Generation of DFT beam grid parameters and precoding matrix for given CSI codebook typeII-rel16 PMI coefficients:

```
    >>> coefs = {
    ...     "i11": "0100",
    ...     "i12": "0001000",
    ...     "i18": "001010011011",
    ...     "i23": "0001110001010111",
    ...     "i15": "",
    ...     "i16": "100101001010100101001010100101001010100101001010",
    ...     "i24": "011010010100",
    ...     "i25": "1111111110111000",
    ...     "i17": "1111100001000011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    ... }
    >>> gen_r16 = rel16.PmiGenerator(comb_inx=6, R=1, N3=18, N1N2=(4, 2), v=4)
    >>> pmi = gen_r16.coef_factory(**coefs)
    >>> pmi.log_summary()
    INFO:nr_csi_pmi.rel16:Coefficients: {'i11': 4, 'i12': 8, 'i23': [1, 12, 5, 7], 'i18': [1, 2, 3, 3], 'i15': 0, 'i16': [2378, 2378, 2378, 2378], 'i24': ['0x60', '0x40', '0x40', '0x80'], 'i25': ['0xf0', '0xf0', '0xb0', '0x80'], 'i17': ['0xc000000000', '0xa000000000', '0x9000000000', '0x9000000000']}
    INFO:nr_csi_pmi.rel16:Beam settings: {'q1q2': (1, 0), 'n1n2': ([2, 0, 1, 2], [0, 1, 1, 1]), 'n3l': [[0, 1, 2, 3, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 5]], 'k1': [1, 12, 5, 7], 'k2': [3, 2, 2, 4], 'c': [15, 15, 11, 8], 'lifs_strongest': [(1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 3, 0)], 'lifs_strongest_xpol': [(1, 5, 0), (2, 6, 0), (3, 7, 0), (4, 7, 0)], 'lifs_other': [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]}
    INFO:nr_csi_pmi.rel16:Coefficients bin: {'i11': '0100', 'i12': '0001000', 'i23': '0001110001010111', 'i18': '001010011011', 'i15': '0', 'i16': '100101001010100101001010100101001010100101001010', 'i24': '011010010100', 'i25': '1111111110111000', 'i17': '1111100001000011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'}
    INFO:nr_csi_pmi.rel16:Coefficients str: [i11 0100, i12 0001000, i18 001010011011, i23 0001110001010111, i15 0, i16 100101001010100101001010100101001010100101001010, i24 011010010100, i25 1111111110111000, i17 C000000000A00000000090000000009000000000]
    >>> pmi.w.for_sb(1)
    array([[ 0.21111187-0.01642076j,  0.20251036-0.01178593j,  0.1622905 -0.02843393j,  0.1077411 +0.j        ],
           [-0.13188061-0.01642076j, -0.1456425 -0.01178593j, -0.18586236-0.02843393j, -0.22559223+0.j        ],
           [ 0.11556853+0.06568306j, -0.09745824+0.16084662j, -0.16084662-0.03588431j, -0.09955278-0.04126484j],
           [-0.20135652-0.06568306j,  0.03588431-0.16084662j,  0.16084662+0.09745824j,  0.20844722+0.08640182j],
           (...)
```

- Get details of an example PMI coefficient (e.g. i17l):

```
    >>> pmi.i17.lifs
           prio_inx  flat_inx  prio
    l i f
    1 0 0         0         0     1
    2 0 0         1        40     2
    3 0 0         2        80     3
    4 0 0         3       120     4
    1 1 0         4         1     5
    2 2 0         9        42    10
    3 3 0        14        83    15
    4 3 0        15       123    16
    >>> pmi.i17.as_bin
    '1111100001000011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    >>> pmi.i17.bin_flat
    '1100000000000000000000000000000000000000101000000000000000000000000000000000000010010000000000000000000000000000000000001001000000000000000000000000000000000000'
    >>> pmi.i17.as_hex
    '0xf843000000000000000000000000000000000000'
    >>> pmi.i17.i172
    '0xa000000000'
```

- Show step by step precoding matrix generation with all the intermediate results:

```
    >>> pmi.log_verbose()
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.base.Vm1m2'>: m1=[np.int64(9), np.int64(1), np.int64(5), np.int64(9)], m2=[np.int64(0), np.int64(4), np.int64(4), np.int64(4)]
    [[ 1.   +0.j     1.   +0.j     1.   +0.j     1.   +0.j     0.   +0.j    0.   +0.j     0.   +0.j     0.   +0.j   ]
     (...)
     [ 0.   -0.j     0.   -0.j    -0.   +0.j     0.   +0.j    -0.383-0.924j  -0.383-0.924j -0.924+0.383j  0.383+0.924j]]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.Y'>:i16l=[2378, 2378, 2378, 2378] (['100101001010', '100101001010', '100101001010', '100101001010']), n3l:[[0, 1, 2, 3, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.I17l'>: 1111100001000011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    i17_flat: 1100000000000000000000000000000000000000101000000000000000000000000000000000000010010000000000000000000000000000000000001001000000000000000000000000000000000000
    i17 lifs:
           prio_inx  flat_inx  prio
    l i f
    1 0 0         0         0     1
    2 0 0         1        40     2
    3 0 0         2        80     3
    4 0 0         3       120     4
    1 1 0         4         1     5
    2 2 0         9        42    10
    3 3 0        14        83    15
    4 3 0        15       123    16
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.I23l'>: 0001110001010111, k1=[1, 12, 5, 7], p1=[0.08838834764831843, 0.5946035575013605, 0.17677669529663687, 0.25]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.I24l'>: 011010010100, k2=[3, 2, 2, 4], p2=[0.25, 0.17677669529663687, 0.17677669529663687, 0.35355339059327373]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.I25l'>: 1111111110111000, c=[15, 15, 11, 8], phi=[np.complex128(0.924-0.383j), np.complex128(0.924-0.383j), np.complex128(-0.383-0.924j), np.complex128(-1+0j)]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.I18l'>: 001010011011 ([1, 2, 3, 3]), lif: [(1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 3, 0)],  flat: [1, 10, 19, 27]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.P1'>: strongest:
           prio_inx  flat_inx  prio
    l i f
    1 1 0         4         1     5
    2 2 0         9        42    10
    3 3 0        14        83    15
    4 3 0        15       123    16
    strongest_xpol:
           prio_inx  flat_inx  prio
    l i f
    1 5 0        20         5    21
    2 6 0        25        46    26
    3 7 0        30        87    31
    4 7 0        31       127    32
    matrix (4, 8):
    [[1.         1.         1.         1.         0.08838835 0.08838835     0.08838835 0.08838835]
     [1.         1.         1.         1.         0.59460356 0.59460356     0.59460356 0.59460356]
     [1.         1.         1.         1.         0.1767767  0.1767767      0.1767767  0.1767767 ]
     [1.         1.         1.         1.         0.25       0.25           0.25       0.25      ]]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.P2'>: lifs:
           prio_inx  flat_inx  prio
    l i f
    1 0 0         0         0     1
    2 0 0         1        40     2
    3 0 0         2        80     3
    4 0 0         3       120     4
    matrix (20, 8) ( (v*M_v, 2L)):
    [[0.25       1.         0.         0.         0.         0.       0.         0.        ]
     (...)
     [0.         0.         0.         0.         0.         0.       0.         0.        ]]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.Phi'>: lifs:
           prio_inx  flat_inx  prio
    l i f
    1 0 0         0         0     1
    2 0 0         1        40     2
    3 0 0         2        80     3
    4 0 0         3       120     4
    matrix (20, 8) ( (v*M_v, 2L)):
    [[ 0.924-0.383j  1.   +0.j     0.   +0.j     0.   +0.j     0.   +0.j     0.   +0.j     0.   +0.j     0.   +0.j   ]
     (...)
     [ 0.   +0.j     0.   +0.j     0.   +0.j     0.   +0.j     0.   +0.j     0.   +0.j     0.   +0.j     0.   +0.j   ]]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.YP2Phitl'>: matrix (72, 8) (v*N3, 2L):
    [[ 0.231     -0.09575j     1.        +0.j          0.        +0.j      0.        +0.j          0.        +0.j          0.        +0.j      0.        +0.j          0.        +0.j        ]
     (...)
     [-0.35355339+0.j          0.        +0.j          0.        +0.j      1.        +0.j          0.        +0.j          0.        +0.j      0.        +0.j          0.        +0.j        ]]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.P1Yp2Phitl'>: matrix (72, 8) (v*N3, 2L):
    [[ 0.231     -0.09575j     1.        +0.j          0.        +0.j      0.        +0.j          0.        +0.j          0.        +0.j      0.        +0.j          0.        +0.j        ]
     (...)
     [-0.35355339+0.j          0.        +0.j          0.        +0.j      1.        +0.j          0.        +0.j          0.        +0.j      0.        +0.j          0.        +0.j        ]]
    INFO:nr_csi_pmi.rel16:Gamma matrix:
    [1.06252906 1.06252906 1.06252906 1.06252906 1.06252906 1.06252906 1.06252906 1.06252906 1.06252906 1.06252906 1.06252906 1.06252906
     (...)
     1.125      1.125      1.125      1.125      1.125      1.125     1.125      1.125      1.125      1.125      1.125      1.125     ]
    INFO:nr_csi_pmi.rel16:<class 'nr_csi_pmi.rel16.W'>: matrix (16, 72) (2*N1*N2, N3*v):
    [[ 0.21111187-0.01642076j  0.21111187-0.01642076j  0.21111187-0.01642076j  ...  0.1077411 +0.j          0.1077411 +0.j   0.1077411 +0.j        ]
     (...)
     [ 0.        +0.j          0.        +0.j          0.        +0.j  ...  0.        +0.j          0.        +0.j   0.        +0.j        ]]
```

- Generation of DFT beam grid parameters and precoding matrix for CSI codebook typeII (rel15) PMI coefficients given as hex encoded X1 and X2:

```
    >>> import logging
    >>> logging.basicConfig(level=logging.INFO)
    >>> from nr_csi_pmi import rel15
    >>> coefs = {
    ...    "N1N2": (4, 2),
    ...    "subbands": "01111111111111111",
    ...    "Npsk": 8,
    ...    "sbAmp": True,
    ...    "L": 2,
    ...    "v": 2,
    ...    "m_l": (3, 4),
    ...    "x1": "0xF7A0A942",
    ...    "x2": "0x58D8D7C94C4EF460C718F11BB10A1AA8A594DF05D35759E07C9566EB14F2B224390A382D1D7049CC"
    ...  }
    >>> gen_rel15 = rel15.PmiGenerator(**coefs)
    >>> pmi = gen_rel15.coef_factory(**coefs)
    >>> pmi.log_summary()
    INFO:nr_csi_pmi.rel15:Coefficients: {'i11': 15, 'i12': 15, 'i131': 1, 'i141': 10, 'i132': 2, 'i142': 161, 'x2': [{'sb': 0, 'i211': 22, 'i212': 108, 'i221': 1, 'i222': 5}, {'sb': 1, 'i211': 52, 'i212': 427, 'i221': 2, 'i222': 5}, {'sb': 2, 'i211': 31, 'i212': 74, 'i221': 1, 'i222': 4}, {'sb': 3, 'i211': 39, 'i212': 259, 'i221': 3, 'i222': 4}, {'sb': 4, 'i211': 19, 'i212': 378, 'i221': 0, 'i222': 6}, {'sb': 5, 'i211': 37, 'i212': 179, 'i221': 1, 'i222': 6}, {'sb': 6, 'i211': 3, 'i212': 56, 'i221': 3, 'i222': 0}, {'sb': 7, 'i211': 44, 'i212': 167, 'i221': 2, 'i222': 2}, {'sb': 8, 'i211': 60, 'i212': 141, 'i221': 3, 'i222': 3}, {'sb': 9, 'i211': 44, 'i212': 274, 'i221': 0, 'i222': 3}, {'sb': 10, 'i211': 4, 'i212': 80, 'i221': 3, 'i222': 2}, {'sb': 11, 'i211': 36, 'i212': 81, 'i221': 3, 'i222': 0}, {'sb': 12, 'i211': 42, 'i212': 82, 'i221': 3, 'i222': 1}, {'sb': 13, 'i211': 11, 'i212': 142, 'i221': 2, 'i222': 7}, {'sb': 14, 'i211': 19, 'i212': 248, 'i221': 0, 'i222': 5}, {'sb': 15, 'i211': 1, 'i212': 78, 'i221': 1, 'i222': 4}]}
    INFO:nr_csi_pmi.rel15:Beam settings: {'q1q2': (3, 3), 'n1n2': ([1, 3], [0, 1]), 'strongest': [1, 2], 'k1': ([0, 1, 2], [2, 4, 1]), 'k2': [([0, 1], [1, 0, 1]), ([1, 0], [1, 0, 1]), ([0, 1], [1, 0, 0]), ([1, 1], [1, 0, 0]), ([0, 0], [1, 1, 0]), ([0, 1], [1, 1, 0]), ([1, 1], [0, 0, 0]), ([1, 0], [0, 1, 0]), ([1, 1], [0, 1, 1]), ([0, 0], [0, 1, 1]), ([1, 1], [0, 1, 0]), ([1, 1], [0, 0, 0]), ([1, 1], [0, 0, 1]), ([1, 0], [1, 1, 1]), ([0, 0], [1, 0, 1]), ([0, 1], [1, 0, 0])], 'c': [([2, 6], [1, 5, 4]), ([6, 4], [6, 5, 3]), ([3, 7], [1, 1, 2]), ([4, 7], [4, 0, 3]), ([2, 3], [5, 7, 2]), ([4, 5], [2, 6, 3]), ([0, 3], [0, 7, 0]), ([5, 4], [2, 4, 7]), ([7, 4], [2, 1, 5]), ([5, 4], [4, 2, 2]), ([0, 4], [1, 2, 0]), ([4, 4], [1, 2, 1]), ([5, 2], [1, 2, 2]), ([1, 3], [2, 1, 6]), ([2, 3], [3, 7, 0]), ([0, 1], [1, 1, 6])], 'subbands': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
    INFO:nr_csi_pmi.rel15:Coefficients bin: {'i11': '1111', 'i12': '01111', 'i131': '01', 'i141': '000001010', 'i132': '10', 'i142': '010100001', 'x2': '01011000110110001101011111001001010011000100111011110100011000001100011100011000111100010001101110110001000010100001101010101000101001011001010011011111000001011101001101010111010110011110000001111100100101010110011011101011000101001111001010110010001001000011100100001010001110000010110100011101011100000100100111001100'}
    INFO:nr_csi_pmi.rel15:Coefficients hex: M1 3 M2 4 PMI [X1 F7A0A942, X2 58D8D7C94C4EF460C718F11BB10A1AA8A594DF05D35759E07C9566EB14F2B224390A382D1D7049CC]
    INFO:nr_csi_pmi.rel15:Coefficients part2 hex: M1 3 M2 4 X 0xf7a0a942b1b1af92989de8c18e31e237621435514b29be0ba6aeb3c0f92acdd629e564487214705a3ae09398
    >>> pmi.w.for_sb(7)
    array([[-1.69030851e-01+0.j        , -3.67354386e-02+0.08737209j],
           [-1.19504812e-01+0.11950481j,  8.77440227e-02+0.03580011j],
           (...)
           [ 4.37959490e-02-0.05047511j,  6.61520720e-02+0.11177967j],
           [-2.76136891e-02-0.01143488j, -1.45730142e-01-0.08020502j]])
```

- Generation of CSI codebook typeII (rel15) PMI coefficients and corresponding precoding matrix for given DFT beam grid parameters:

```
    >>> beam_settings = {
    ...    'q1q2': (3, 3), 'n1n2': ([1, 3], [0, 1]),
    ...    'strongest': [2, 0],
    ...    'k1': ([0, 1, 2], [2, 4, 1]),
    ...    'k2': [([0, 1], [1, 0, 1]), ([1, 0], [1, 0, 1]), ([0, 1], [1, 0, 0]), ([1, 1], [1, 0, 0])] * 4,
    ...    'c': [([2, 6], [1, 5, 4]), ([6, 4], [6, 5, 3]), ([3, 7], [1, 1, 2]), ([4, 7], [4, 0, 3])] * 4,
    ... }
    >>> gen_rel15 = rel15.PmiGenerator(N1N2=(4,2), subbands="01111111111111111", Npsk=8, sbAmp=True, L=2)
    >>> pmi = gen_rel15.beam_factory(**beam_settings)
    >>> pmi.log_summary()
    INFO:nr_csi_pmi.rel15:Coefficients: {'i11': 15, 'i12': 15, 'i131': 2, 'i141': 10, 'i132': 0, 'i142': 161, 'x2': [{'sb': 0, 'i211': 22, 'i212': 108, 'i221': 1, 'i222': 5}, {'sb': 1, 'i211': 52, 'i212': 427, 'i221': 2, 'i222': 5}, {'sb': 2, 'i211': 31, 'i212': 74, 'i221': 1, 'i222': 4}, {'sb': 3, 'i211': 39, 'i212': 259, 'i221': 3, 'i222': 4}, {'sb': 4, 'i211': 22, 'i212': 108, 'i221': 1, 'i222': 5}, {'sb': 5, 'i211': 52, 'i212': 427, 'i221': 2, 'i222': 5}, {'sb': 6, 'i211': 31, 'i212': 74, 'i221': 1, 'i222': 4}, {'sb': 7, 'i211': 39, 'i212': 259, 'i221': 3, 'i222': 4}, {'sb': 8, 'i211': 22, 'i212': 108, 'i221': 1, 'i222': 5}, {'sb': 9, 'i211': 52, 'i212': 427, 'i221': 2, 'i222': 5}, {'sb': 10, 'i211': 31, 'i212': 74, 'i221': 1, 'i222': 4}, {'sb': 11, 'i211': 39, 'i212': 259, 'i221': 3, 'i222': 4}, {'sb': 12, 'i211': 22, 'i212': 108, 'i221': 1, 'i222': 5}, {'sb': 13, 'i211': 52, 'i212': 427, 'i221': 2, 'i222': 5}, {'sb': 14, 'i211': 31, 'i212': 74, 'i221': 1, 'i222': 4}, {'sb': 15, 'i211': 39, 'i212': 259, 'i221': 3, 'i222': 4}]}
    INFO:nr_csi_pmi.rel15:Beam settings: {'q1q2': (3, 3), 'n1n2': ([1, 3], [0, 1]), 'strongest': [2, 0], 'k1': ([0, 1, 2], [2, 4, 1]), 'k2': [([0, 1], [1, 0, 1]), ([1, 0], [1, 0, 1]), ([0, 1], [1, 0, 0]), ([1, 1], [1, 0, 0]), ([0, 1], [1, 0, 1]), ([1, 0], [1, 0, 1]), ([0, 1], [1, 0, 0]), ([1, 1], [1, 0, 0]), ([0, 1], [1, 0, 1]), ([1, 0], [1, 0, 1]), ([0, 1], [1, 0, 0]), ([1, 1], [1, 0, 0]), ([0, 1], [1, 0, 1]), ([1, 0], [1, 0, 1]), ([0, 1], [1, 0, 0]), ([1, 1], [1, 0, 0])], 'c': [([2, 6], [1, 5, 4]), ([6, 4], [6, 5, 3]), ([3, 7], [1, 1, 2]), ([4, 7], [4, 0, 3]), ([2, 6], [1, 5, 4]), ([6, 4], [6, 5, 3]), ([3, 7], [1, 1, 2]), ([4, 7], [4, 0, 3]), ([2, 6], [1, 5, 4]), ([6, 4], [6, 5, 3]), ([3, 7], [1, 1, 2]), ([4, 7], [4, 0, 3]), ([2, 6], [1, 5, 4]), ([6, 4], [6, 5, 3]), ([3, 7], [1, 1, 2]), ([4, 7], [4, 0, 3])], 'subbands': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
    INFO:nr_csi_pmi.rel15:Coefficients bin: {'i11': '1111', 'i12': '01111', 'i131': '10', 'i141': '000001010', 'i132': '00', 'i142': '010100001', 'x2': '01011000110110001101011111001001010011000101100011011000110101111100100101001100010110001101100011010111110010010100110001011000110110001101011111001001010011001101001101010111010110011110000001111100110100110101011101011001111000000111110011010011010101110101100111100000011111001101001101010111010110011110000001111100'}
    INFO:nr_csi_pmi.rel15:Coefficients hex: M1 3 M2 4 PMI [X1 F7C0A142, X2 58D8D7C94C58D8D7C94C58D8D7C94C58D8D7C94CD35759E07CD35759E07CD35759E07CD35759E07C]
    INFO:nr_csi_pmi.rel15:Coefficients part2 hex: M1 3 M2 4 X 0xf7c0a142b1b1af9298b1b1af9298b1b1af9298b1b1af9299a6aeb3c0f9a6aeb3c0f9a6aeb3c0f9a6aeb3c0f8
    >>> pmi.w.for_sb(5)
    array([[-0.01532848+0.j        , -0.09313327-0.12606582j],
           [-0.01083724+0.01083724j,  0.10840712-0.02328332j],
           [-0.01416352+0.00587081j,  0.10911201+0.01995542j],
           [-0.00587081+0.01416352j, -0.15215486+0.03777193j],
           [-0.01083724+0.01083724j, -0.15497376-0.02328332j],
           [ 0.        +0.01532848j,  0.06020071-0.09313327j],
           [-0.00587081+0.01416352j,  0.0912955 -0.06299827j],
           [ 0.00587081+0.01416352j, -0.08081478+0.13433835j],
           [ 0.03065697-0.17342199j, -0.0550286 +0.07831543j],
           [ 0.14428383+0.10093487j,  0.01646379-0.12720181j],
           [ 0.09474766+0.1485003j ,  0.06388568-0.11127712j],
           [-0.1485003 -0.09474766j,  0.0512875 +0.08084123j],
           [-0.10093487-0.14428383j,  0.01646379+0.09427423j],
           [ 0.17342199-0.03065697j, -0.07831543-0.10160226j],
           [ 0.17198354+0.03809359j, -0.03344979-0.1238753j ],
           [-0.17198354+0.03809359j,  0.09343941+0.02085161j]])
```

## Restrictions

- The codebook subset restriction and corresponding maximum amplitude (rel15) or maximum average amplitude (r16) restrictions for indicated vector groups are currently not supported,
